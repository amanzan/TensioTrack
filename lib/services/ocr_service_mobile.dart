import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:google_generative_ai/google_generative_ai.dart';
import 'package:http/http.dart' as http;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'blood_pressure_digit_reader.dart';
import 'ocr_service.dart';

/// Retorna la instancia de servicio OCR móvil.
OcrService getOcrService() => GeminiOcrService();

class GeminiOcrService implements OcrService {
  /// Clave API de Gemini.
  static const _geminiApiKey = String.fromEnvironment('GEMINI_API_KEY');

  /// Clave API de Groq.
  static const _groqApiKey = String.fromEnvironment('GROQ_API_KEY');

  /// Token de GitHub Models con permiso models:read.
  static const _githubModelsToken = String.fromEnvironment(
    'GITHUB_MODELS_TOKEN',
  );
  static const _githubModelsModel = String.fromEnvironment(
    'GITHUB_MODELS_MODEL',
    defaultValue: 'openai/gpt-4o-mini',
  );

  /// Timeout máximo para las peticiones a proveedores cloud.
  static const _timeout = Duration(seconds: 60);

  /// FLAG TEMPORAL PARA PRUEBAS
  /// Si es true, la app utiliza el motor offline seleccionado.
  /// Si es false, usa Gemini en la nube como primera opción y OCR local como respaldo.
  static bool forceOfflineOcr = const bool.fromEnvironment('FORCE_OFFLINE_OCR');

  /// FLAG TEMPORAL PARA PRUEBAS
  /// Si es true, omite Gemini y realiza los intentos cloud solo con Groq.
  static bool forceGroqOcr = const bool.fromEnvironment('FORCE_GROQ_OCR');

  /// FLAG TEMPORAL PARA PRUEBAS
  /// Si es true, omite Gemini/Groq y prueba solo GitHub Models.
  static bool forceGithubOcr = const bool.fromEnvironment('FORCE_GITHUB_OCR');

  static DateTime? _geminiQuotaBlockedUntil;

  final _digitReader = BloodPressureDigitReader();

  @override
  Future<OcrResult?> recognizePressure(
    String imagePath,
    Uint8List imageBytes,
  ) async {
    if (forceOfflineOcr) {
      debugPrint(
        'TensioTrack: forceOfflineOcr activo. Ejecutando OCR offline...',
      );
      return _recognizeWithOfflineCascade(imagePath, imageBytes);
    }

    try {
      debugPrint(
        forceGithubOcr
            ? 'TensioTrack: Modo cloud forzado a solo GitHub Models.'
            : forceGroqOcr
            ? 'TensioTrack: Modo cloud forzado a solo Groq.'
            : 'TensioTrack: Modo cloud Gemini -> GitHub Models -> Gemini -> GitHub Models -> Groq -> Groq.',
      );
      final cloudResult = await _recognizeWithCloudProvidersWithRetries(
        imageBytes,
      );
      if (cloudResult != null) return cloudResult;

      debugPrint(
        'TensioTrack: Ningún proveedor cloud devolvió lectura. Usando fallback OCR offline...',
      );
      return await _recognizeWithOfflineCascade(imagePath, imageBytes);
    } on TimeoutException {
      debugPrint('TensioTrack Cloud TIMEOUT: Conmutando a OCR offline...');
      return await _recognizeWithOfflineCascade(imagePath, imageBytes);
    } catch (e) {
      debugPrint('TensioTrack Cloud ERROR ($e): Conmutando a OCR offline...');
      return await _recognizeWithOfflineCascade(imagePath, imageBytes);
    }
  }

  Future<OcrResult?> _recognizeWithOfflineCascade(
    String imagePath,
    Uint8List imageBytes,
  ) async {
    debugPrint(
      'TensioTrack: motor offline seleccionado: '
      '${OfflineOcrConfig.engine.label}.',
    );

    return switch (OfflineOcrConfig.engine) {
      OfflineOcrEngine.hybrid => _recognizeWithSelectedHybrid(
        imagePath,
        imageBytes,
      ),
      OfflineOcrEngine.yolo => _recognizeWithSelectedYolo(imageBytes),
    };
  }

  Future<OcrResult?> _recognizeWithSelectedHybrid(
    String imagePath,
    Uint8List imageBytes,
  ) async {
    try {
      debugPrint('TensioTrack: Ejecutando híbrido YOLO + CNN local...');
      return await _recognizeWithHybrid(imagePath, imageBytes);
    } catch (e) {
      debugPrint('TensioTrack: Falló Híbrido en modo offline ($e).');
      return null;
    }
  }

  Future<OcrResult?> _recognizeWithHybrid(
    String imagePath,
    Uint8List imageBytes,
  ) async {
    debugPrint('TensioTrack: Inicializando pipeline híbrido (YOLO + CNN)...');

    // 1. Ejecutar YOLO para obtener la localización de las cajas
    final yoloResult = await _digitReader.readFromImageBytes(imageBytes);
    if (yoloResult == null || yoloResult.detections.isEmpty) {
      debugPrint('TensioTrack Híbrido: YOLO no detectó ningún dígito.');
      return null;
    }

    // 2. Decodificar la imagen original
    final Image originalImage;
    double imageWidth = yoloResult.imageWidth.toDouble();
    double imageHeight = yoloResult.imageHeight.toDouble();
    try {
      final codec = await instantiateImageCodec(imageBytes);
      final frame = await codec.getNextFrame();
      originalImage = frame.image;
    } catch (e) {
      debugPrint('TensioTrack Híbrido ERROR: No se pudo decodificar la imagen: $e');
      return null;
    }

    // 3. Inicializar intérprete de la CNN
    final cnnInterpreter = await Interpreter.fromAsset('digit_classifier.tflite');

    // 4. Clasificar cada dígito con la CNN
    final updatedDetections = <DigitDetection>[];

    for (final det in yoloResult.detections) {
      try {
        final dh = det.y2 - det.y1;
        final padX = dh * 0.07;
        final padY = dh * 0.12;
        
        final x1 = (det.x1 - padX).clamp(0.0, imageWidth);
        final y1 = (det.y1 - padY).clamp(0.0, imageHeight);
        final x2 = (det.x2 + padX).clamp(0.0, imageWidth);
        final y2 = (det.y2 + padY).clamp(0.0, imageHeight);
        
        final cropRect = Rect.fromLTRB(x1, y1, x2, y2);
        
        // Binarizar en resolución original y luego redimensionar a 28x28
        final binBytes = await _preprocessDigitForCnn(originalImage, cropRect);
        if (binBytes == null) continue;
        
        // Preparar entrada [1, 28, 28, 1]
        final input = Float32List(1 * 28 * 28 * 1);
        for (int i = 0; i < 28 * 28; i++) {
          final r = binBytes[i * 4];
          input[i] = (255 - r) / 255.0; // Invertir: fondo 0, texto 1
        }
        
        final output = List.generate(
          1,
          (_) => List<double>.filled(10, 0.0, growable: false),
          growable: false,
        );
        
        cnnInterpreter.run(input.buffer, output);
        
        // Ensamble probabilístico: Combinar la predicción de la CNN con la de YOLO
        // Le damos un peso al prior de YOLO y el restante a la CNN especializada.
        final double yoloWeight = double.tryParse(const String.fromEnvironment('YOLO_WEIGHT')) ?? 0.50;
        final combinedScores = List<double>.generate(10, (c) {
          final cnnProb = output[0][c];
          final yoloPrior = (c == det.digit) ? det.confidence : 0.0;
          return cnnProb * (1.0 - yoloWeight) + yoloPrior * yoloWeight;
        });

        int predDigit = 0;
        double maxScore = combinedScores[0];
        for (int c = 1; c < 10; c++) {
          if (combinedScores[c] > maxScore) {
            maxScore = combinedScores[c];
            predDigit = c;
          }
        }
        
        updatedDetections.add(
          DigitDetection(
            digit: predDigit,
            confidence: maxScore,
            x1: det.x1,
            y1: det.y1,
            x2: det.x2,
            y2: det.y2,
          ),
        );
      } catch (e) {
        debugPrint('Error clasificando dígito individual: $e');
      }
    }

    cnnInterpreter.close();

    if (updatedDetections.isEmpty) return null;

    // 5. Agrupar en filas usando la lógica de YOLO
    final rows = _groupRows(updatedDetections);
    if (rows.length < 2) {
      debugPrint('TensioTrack Híbrido: Detecciones insuficientes para SYS/DIA (filas=${rows.length}).');
      return null;
    }

    final rowValues = rows
        .map((row) => int.tryParse(row.map((d) => d.digit).join()))
        .whereType<int>()
        .toList(growable: false);

    if (rowValues.length < 2) return null;

    int sysVal = rowValues[0];
    int diaVal = rowValues[1];
    if (sysVal < diaVal) {
      final temp = sysVal;
      sysVal = diaVal;
      diaVal = temp;
    }

    final firstTwoRows = rows.take(2).expand((row) => row);
    final confidence =
        firstTwoRows.map((d) => d.confidence).reduce((a, b) => a + b) /
        firstTwoRows.length;

    debugPrint('TensioTrack Híbrido: Extraído SYS=$sysVal, DIA=$diaVal (conf=${confidence.toStringAsFixed(2)})');

    return OcrResult(
      systolic: sysVal,
      diastolic: diaVal,
      systolicBox: _rowBoundingBox(rows[0]),
      diastolicBox: _rowBoundingBox(rows[1]),
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      confidence: confidence,
      engineName: 'Híbrido YOLO + CNN (Offline)',
    );
  }


  /// Realiza binarización adaptativa local (Bradley-Roth) sobre la resolución original
  /// y luego redimensiona a 28x28, para alimentar el clasificador CNN.
  Future<Uint8List?> _preprocessDigitForCnn(Image originalImage, Rect cropRect) async {
    final double width = cropRect.width;
    final double height = cropRect.height;
    if (width <= 0 || height <= 0) return null;

    final recorder = PictureRecorder();
    final canvas = Canvas(recorder);
    canvas.drawImageRect(
      originalImage,
      cropRect,
      Rect.fromLTWH(0, 0, width, height),
      Paint()..filterQuality = FilterQuality.high,
    );
    final picture = recorder.endRecording();
    final cropped = await picture.toImage(width.toInt(), height.toInt());

    final byteData = await cropped.toByteData(format: ImageByteFormat.rawRgba);
    if (byteData == null) return null;
    final pixels = byteData.buffer.asUint8List();
    final w = cropped.width;
    final h = cropped.height;

    // 1. Convertir a escala de grises
    final grays = Uint8List(w * h);
    for (int i = 0; i < pixels.length; i += 4) {
      int r = pixels[i];
      int g = pixels[i + 1];
      int b = pixels[i + 2];
      grays[i ~/ 4] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0, 255);
    }

    // 2. Imagen integral
    final integral = Int32List(w * h);
    for (int y = 0; y < h; y++) {
      int rowSum = 0;
      for (int x = 0; x < w; x++) {
        rowSum += grays[y * w + x];
        if (y == 0) {
          integral[y * w + x] = rowSum;
        } else {
          integral[y * w + x] = integral[(y - 1) * w + x] + rowSum;
        }
      }
    }

    // 3. Umbral adaptativo local (Bradley-Roth)
    // Emula cv2.adaptiveThreshold con blockSize=25 y C=9
    final int s = 25.clamp(7, math.min(w, h));
    const double t = 0.08;

    final binarizedGrays = Uint8List(w * h);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int x1 = (x - s ~/ 2).clamp(0, w - 1);
        int x2 = (x + s ~/ 2).clamp(0, w - 1);
        int y1 = (y - s ~/ 2).clamp(0, h - 1);
        int y2 = (y + s ~/ 2).clamp(0, h - 1);

        int count = (x2 - x1 + 1) * (y2 - y1 + 1);
        int sum = integral[y2 * w + x2];
        if (x1 > 0) sum -= integral[y2 * w + (x1 - 1)];
        if (y1 > 0) sum -= integral[(y1 - 1) * w + x2];
        if (x1 > 0 && y1 > 0) sum += integral[(y1 - 1) * w + (x1 - 1)];

        int gray = grays[y * w + x];
        binarizedGrays[y * w + x] = (gray * count < sum * (1.0 - t)) ? 0 : 255;
      }
    }

    // 4. Convertir binarizado a formato RGBA
    final binarizedRgba = Uint8List(w * h * 4);
    for (int i = 0; i < w * h; i++) {
      final val = binarizedGrays[i];
      binarizedRgba[i * 4] = val;
      binarizedRgba[i * 4 + 1] = val;
      binarizedRgba[i * 4 + 2] = val;
      binarizedRgba[i * 4 + 3] = 255;
    }

    // 5. Instanciar Image de UI a partir de los bytes binarizados
    final completer = Completer<Image>();
    decodeImageFromPixels(
      binarizedRgba,
      w,
      h,
      PixelFormat.rgba8888,
      completer.complete,
    );
    final binImage = await completer.future;

    // 6. Redimensionar a 28x28 usando Canvas
    final recorderResize = PictureRecorder();
    final canvasResize = Canvas(recorderResize);
    canvasResize.drawImageRect(
      binImage,
      Rect.fromLTWH(0, 0, w.toDouble(), h.toDouble()),
      Rect.fromLTWH(0, 0, 28, 28),
      Paint()..filterQuality = FilterQuality.high,
    );
    final pictureResize = recorderResize.endRecording();
    final resizedImage = await pictureResize.toImage(28, 28);

    final resizedByteData = await resizedImage.toByteData(format: ImageByteFormat.rawRgba);
    if (resizedByteData == null) return null;
    return resizedByteData.buffer.asUint8List();
  }

  List<List<DigitDetection>> _groupRows(List<DigitDetection> detections) {
    if (detections.isEmpty) return const [];

    final sorted = [...detections]..sort((a, b) => a.cy.compareTo(b.cy));
    final avgHeight =
        sorted.map((d) => d.height).reduce((a, b) => a + b) / sorted.length;
    final yThreshold = avgHeight * 0.6; // rowYThresholdFactor = 0.6
    final rows = <List<DigitDetection>>[];

    for (final detection in sorted) {
      var placed = false;

      for (final row in rows) {
        final rowCy = _averageCy(row);
        if ((detection.cy - rowCy).abs() < yThreshold) {
          row.add(detection);
          placed = true;
          break;
        }
      }

      if (!placed) {
        rows.add([detection]);
      }
    }

    for (final row in rows) {
      row.sort((a, b) => a.cx.compareTo(b.cx));
    }
    rows.sort((a, b) => _averageCy(a).compareTo(_averageCy(b)));
    return rows;
  }

  double _averageCy(List<DigitDetection> row) {
    return row.map((d) => d.cy).reduce((a, b) => a + b) / row.length;
  }



  Future<OcrResult?> _recognizeWithSelectedYolo(Uint8List imageBytes) async {
    try {
      return await _recognizeWithYoloDigits(imageBytes);
    } catch (e) {
      debugPrint('TensioTrack: Falló YOLOv8 TFLite offline ($e).');
      return null;
    }
  }


  Future<OcrResult?> _recognizeWithYoloDigits(Uint8List imageBytes) async {
    debugPrint('TensioTrack YOLOv8 TFLite: ejecutando detección de dígitos...');
    final result = await _digitReader.readFromImageBytes(imageBytes);
    if (result == null) {
      debugPrint('TensioTrack YOLOv8 TFLite: sin lectura válida.');
      return null;
    }

    debugPrint(
      'TensioTrack YOLOv8 TFLite RESULTADO: '
      'SYS=${result.systolic}, DIA=${result.diastolic}, '
      'filas=${result.rowValues}, '
      'detecciones=${result.detections.length}, '
      'conf=${result.confidence.toStringAsFixed(2)}',
    );

    return OcrResult(
      systolic: result.systolic,
      diastolic: result.diastolic,
      systolicBox: _rowBoundingBox(result.rows[0]),
      diastolicBox: _rowBoundingBox(result.rows[1]),
      imageWidth: result.imageWidth.toDouble(),
      imageHeight: result.imageHeight.toDouble(),
      confidence: result.confidence,
      engineName: 'YOLOv8 TFLite dígitos (Offline)',
    );
  }

  Rect? _rowBoundingBox(List<DigitDetection> row) {
    if (row.isEmpty) return null;

    var left = row.first.x1;
    var top = row.first.y1;
    var right = row.first.x2;
    var bottom = row.first.y2;

    for (final detection in row.skip(1)) {
      left = math.min(left, detection.x1);
      top = math.min(top, detection.y1);
      right = math.max(right, detection.x2);
      bottom = math.max(bottom, detection.y2);
    }

    return Rect.fromLTRB(left, top, right, bottom);
  }

  Future<OcrResult?> _recognizeWithCloudProvidersWithRetries(
    Uint8List imageBytes,
  ) async {
    Object? lastError;
    final providerPlan = _cloudProviderPlan();

    for (var index = 0; index < providerPlan.length; index++) {
      final attempt = index + 1;
      final provider = providerPlan[index];

      try {
        debugPrint(
          'TensioTrack ${provider.label}: intento $attempt/${providerPlan.length}...',
        );
        final result = switch (provider) {
          _CloudOcrProvider.gemini => await _recognizeWithGemini(imageBytes),
          _CloudOcrProvider.github => await _recognizeWithGithub(imageBytes),
          _CloudOcrProvider.groq => await _recognizeWithGroq(imageBytes),
        };
        if (result != null) return result;
      } catch (e) {
        lastError = e;
        if (provider == _CloudOcrProvider.gemini && _isGeminiQuotaError(e)) {
          _rememberGeminiQuotaCooldown(e);
        }
        debugPrint(
          'TensioTrack ${provider.label}: intento $attempt/${providerPlan.length} falló ($e).',
        );
      }

      if (attempt < providerPlan.length) {
        await Future<void>.delayed(Duration(milliseconds: 700 * attempt));
      }
    }

    if (lastError != null) throw lastError;
    return null;
  }

  List<_CloudOcrProvider> _cloudProviderPlan() {
    if (forceGithubOcr) {
      return const [_CloudOcrProvider.github, _CloudOcrProvider.github];
    }
    if (forceGroqOcr) {
      return const [_CloudOcrProvider.groq, _CloudOcrProvider.groq];
    }

    if (_isGeminiQuotaBlocked()) {
      debugPrint(
        'TensioTrack Gemini: cuota agotada temporalmente. Usando GitHub Models -> GitHub Models -> Groq -> Groq.',
      );
      return const [
        _CloudOcrProvider.github,
        _CloudOcrProvider.github,
        _CloudOcrProvider.groq,
        _CloudOcrProvider.groq,
      ];
    }

    return const [
      _CloudOcrProvider.gemini,
      _CloudOcrProvider.github,
      _CloudOcrProvider.gemini,
      _CloudOcrProvider.github,
      _CloudOcrProvider.groq,
      _CloudOcrProvider.groq,
    ];
  }

  bool _isGeminiQuotaBlocked() {
    final blockedUntil = _geminiQuotaBlockedUntil;
    if (blockedUntil == null) return false;

    if (DateTime.now().isAfter(blockedUntil)) {
      _geminiQuotaBlockedUntil = null;
      return false;
    }

    return true;
  }

  bool _isGeminiQuotaError(Object error) {
    final errorText = error.toString().toLowerCase();
    return errorText.contains('quota exceeded') ||
        errorText.contains('resource_exhausted') ||
        errorText.contains('current quota');
  }

  void _rememberGeminiQuotaCooldown(Object error) {
    final errorText = error.toString();
    final retryMatch = RegExp(
      r'retry in ([0-9]+(?:\.[0-9]+)?)s',
      caseSensitive: false,
    ).firstMatch(errorText);
    final retrySeconds = retryMatch == null
        ? 60.0
        : double.tryParse(retryMatch.group(1)!) ?? 60.0;

    _geminiQuotaBlockedUntil = DateTime.now().add(
      Duration(milliseconds: (retrySeconds * 1000).ceil()),
    );
    debugPrint(
      'TensioTrack Gemini: cuota agotada. Se evitará Gemini durante ${retrySeconds.toStringAsFixed(1)}s.',
    );
  }




  /// OCR en la nube usando Gemini Vision API
  Future<OcrResult?> _recognizeWithGemini(Uint8List imageBytes) async {
    if (_geminiApiKey.isEmpty) {
      debugPrint(
        'TensioTrack: API key de Gemini no configurada. '
        'Ejecuta con: flutter run --dart-define-from-file=.env.json',
      );
      return null;
    }

    debugPrint(
      'TensioTrack Gemini: Enviando imagen '
      '(${(imageBytes.length / 1024).toStringAsFixed(0)} KB) a Gemini Cloud...',
    );

    final model = GenerativeModel(
      model: 'gemini-2.5-flash-lite',
      apiKey: _geminiApiKey,
    );

    const prompt =
        '''Analyze this photo of a blood pressure monitor (tensiometer).
Read the numeric values shown on the LCD screen display.

The LCD screen typically shows three values:
- Systolic pressure (SYS): usually the top and largest number
- Diastolic pressure (DIA): usually below SYS, second largest
- Pulse rate (PULSE): usually the smallest number at the bottom

Respond ONLY with three integers separated by commas in this exact format:
systolic,diastolic,pulse

Example responses:
136,85,72
120,80,65

If you cannot read a value clearly, use 0 for that value.
Do NOT include any other text, explanation, units, or formatting.''';

    final content = Content.multi([
      TextPart(prompt),
      DataPart('image/jpeg', imageBytes),
    ]);

    final response = await model.generateContent([content]).timeout(_timeout);
    final text = response.text?.trim();

    debugPrint('TensioTrack Gemini respuesta: "$text"');

    if (text == null || text.isEmpty) return null;

    int sys = 0, dia = 0;
    final parts = text.split(',');

    if (parts.length >= 2) {
      sys = int.tryParse(parts[0].trim()) ?? 0;
      dia = int.tryParse(parts[1].trim()) ?? 0;
    } else {
      // Fallback de extracción numérica clásica
      final numbers = RegExp(r'\d+')
          .allMatches(text)
          .map((m) => int.parse(m.group(0)!))
          .where((n) => n >= 30 && n <= 250)
          .toList();

      if (numbers.length >= 2) {
        numbers.sort((a, b) => b.compareTo(a));
        sys = numbers[0];
        dia = numbers[1];
      } else if (numbers.length == 1) {
        if (numbers[0] >= 95) {
          sys = numbers[0];
        } else {
          dia = numbers[0];
        }
      }
    }

    if (sys == 0 && dia == 0) return null;

    debugPrint('TensioTrack Gemini RESULTADO: SYS=$sys, DIA=$dia');

    return _buildCloudResult(
      imageBytes: imageBytes,
      systolic: sys,
      diastolic: dia,
      confidence: (sys > 0 && dia > 0) ? 0.95 : 0.6,
      engineName: 'Gemini Vision (Cloud)',
    );
  }

  /// OCR en la nube usando Groq + Llama Vision.
  Future<OcrResult?> _recognizeWithGroq(Uint8List imageBytes) async {
    if (_groqApiKey.isEmpty) {
      debugPrint(
        'TensioTrack: API key de Groq no configurada. '
        'Ejecuta con: flutter run --dart-define-from-file=.env.json',
      );
      return null;
    }

    debugPrint(
      'TensioTrack Groq: Enviando imagen '
      '(${(imageBytes.length / 1024).toStringAsFixed(0)} KB) a Groq Cloud...',
    );

    final response = await http
        .post(
          Uri.parse('https://api.groq.com/openai/v1/chat/completions'),
          headers: {
            'Authorization': 'Bearer $_groqApiKey',
            'Content-Type': 'application/json',
          },
          body: jsonEncode({
            'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
            'max_tokens': 50,
            'messages': [
              {
                'role': 'user',
                'content': [
                  {
                    'type': 'image_url',
                    'image_url': {
                      'url':
                          'data:${_detectImageMimeType(imageBytes)};base64,${base64Encode(imageBytes)}',
                    },
                  },
                  {
                    'type': 'text',
                    'text':
                        'This is a photo of a blood pressure monitor. '
                        'Return ONLY a JSON object like {"systolic": 120, "diastolic": 80}. '
                        'The systolic is always the larger number, diastolic the smaller. '
                        'No explanation, no markdown, just the JSON.',
                  },
                ],
              },
            ],
          }),
        )
        .timeout(_timeout);

    if (response.statusCode != 200) {
      debugPrint(
        'TensioTrack Groq ERROR: ${response.statusCode} ${response.body}',
      );
      return null;
    }

    final responseJson = jsonDecode(response.body) as Map<String, dynamic>;
    final choices = responseJson['choices'] as List<dynamic>?;
    final message = choices?.isNotEmpty == true
        ? choices!.first['message'] as Map<String, dynamic>?
        : null;
    final content = message?['content'] as String?;

    debugPrint('TensioTrack Groq respuesta: "$content"');

    if (content == null || content.trim().isEmpty) return null;
    final values = _parseCloudBpResponse(content);
    if (values == null) return null;

    debugPrint(
      'TensioTrack Groq RESULTADO: SYS=${values.systolic}, DIA=${values.diastolic}',
    );

    return _buildCloudResult(
      imageBytes: imageBytes,
      systolic: values.systolic,
      diastolic: values.diastolic,
      confidence: 0.92,
      engineName: 'Groq Llama Vision (Cloud)',
    );
  }

  /// OCR en la nube usando GitHub Models multimodal.
  Future<OcrResult?> _recognizeWithGithub(Uint8List imageBytes) async {
    if (_githubModelsToken.isEmpty) {
      debugPrint(
        'TensioTrack: token de GitHub Models no configurado. '
        'Ejecuta con: flutter run --dart-define-from-file=.env.json',
      );
      return null;
    }

    debugPrint(
      'TensioTrack GitHub Models: Enviando imagen '
      '(${(imageBytes.length / 1024).toStringAsFixed(0)} KB) a $_githubModelsModel...',
    );

    final response = await http
        .post(
          Uri.parse('https://models.github.ai/inference/chat/completions'),
          headers: {
            'Accept': 'application/vnd.github+json',
            'Authorization': 'Bearer $_githubModelsToken',
            'Content-Type': 'application/json',
            'X-GitHub-Api-Version': '2026-03-10',
          },
          body: jsonEncode({
            'model': _githubModelsModel,
            'temperature': 0,
            'max_tokens': 60,
            'response_format': {'type': 'json_object'},
            'messages': [
              {
                'role': 'user',
                'content': [
                  {
                    'type': 'image_url',
                    'image_url': {
                      'url':
                          'data:${_detectImageMimeType(imageBytes)};base64,${base64Encode(imageBytes)}',
                    },
                  },
                  {
                    'type': 'text',
                    'text':
                        'This is a photo of a blood pressure monitor. '
                        'Read the LCD numbers and return ONLY JSON in this exact shape: '
                        '{"systolic": 120, "diastolic": 80, "pulse": 65}. '
                        'Use 0 for pulse if you cannot read it. '
                        'The systolic is normally the larger pressure number and diastolic the smaller. '
                        'No explanation, no markdown.',
                  },
                ],
              },
            ],
          }),
        )
        .timeout(_timeout);

    if (response.statusCode != 200) {
      debugPrint(
        'TensioTrack GitHub Models ERROR: ${response.statusCode} ${response.body}',
      );
      return null;
    }

    final responseJson = jsonDecode(response.body) as Map<String, dynamic>;
    final choices = responseJson['choices'] as List<dynamic>?;
    final message = choices?.isNotEmpty == true
        ? choices!.first['message'] as Map<String, dynamic>?
        : null;
    final content = message?['content'] as String?;

    debugPrint('TensioTrack GitHub Models respuesta: "$content"');

    if (content == null || content.trim().isEmpty) return null;
    final values = _parseCloudBpResponse(content);
    if (values == null) return null;

    debugPrint(
      'TensioTrack GitHub Models RESULTADO: SYS=${values.systolic}, DIA=${values.diastolic}',
    );

    return _buildCloudResult(
      imageBytes: imageBytes,
      systolic: values.systolic,
      diastolic: values.diastolic,
      confidence: 0.93,
      engineName: 'GitHub Models ($_githubModelsModel)',
    );
  }

  _PressureValues? _parseCloudBpResponse(String content) {
    try {
      final clean = content.replaceAll(RegExp(r'```json|```'), '').trim();
      final decoded = jsonDecode(clean);
      if (decoded is! Map<String, dynamic>) return null;

      var systolic = (decoded['systolic'] as num?)?.toInt();
      var diastolic = (decoded['diastolic'] as num?)?.toInt();
      if (systolic == null || diastolic == null) return null;
      if (systolic < 40 || systolic > 250) return null;
      if (diastolic < 40 || diastolic > 250) return null;

      if (systolic < diastolic) {
        final temp = systolic;
        systolic = diastolic;
        diastolic = temp;
      }

      return _PressureValues(systolic: systolic, diastolic: diastolic);
    } catch (_) {
      return null;
    }
  }

  Future<OcrResult> _buildCloudResult({
    required Uint8List imageBytes,
    required int systolic,
    required int diastolic,
    required double confidence,
    required String engineName,
  }) async {
    double imageWidth = 1000, imageHeight = 1000;
    try {
      final codec = await instantiateImageCodec(imageBytes);
      final frame = await codec.getNextFrame();
      imageWidth = frame.image.width.toDouble();
      imageHeight = frame.image.height.toDouble();
    } catch (_) {}

    return OcrResult(
      systolic: systolic,
      diastolic: diastolic,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      confidence: confidence,
      engineName: engineName,
    );
  }

  String _detectImageMimeType(Uint8List bytes) {
    if (bytes.length >= 4 &&
        bytes[0] == 0x89 &&
        bytes[1] == 0x50 &&
        bytes[2] == 0x4E &&
        bytes[3] == 0x47) {
      return 'image/png';
    }
    if (bytes.length >= 3 &&
        bytes[0] == 0xFF &&
        bytes[1] == 0xD8 &&
        bytes[2] == 0xFF) {
      return 'image/jpeg';
    }
    if (bytes.length >= 12 &&
        bytes[0] == 0x52 &&
        bytes[1] == 0x49 &&
        bytes[2] == 0x46 &&
        bytes[3] == 0x46 &&
        bytes[8] == 0x57 &&
        bytes[9] == 0x45 &&
        bytes[10] == 0x42 &&
        bytes[11] == 0x50) {
      return 'image/webp';
    }
    return 'image/jpeg';
  }
}

enum _CloudOcrProvider {
  gemini('Gemini'),
  github('GitHub Models'),
  groq('Groq');

  const _CloudOcrProvider(this.label);

  final String label;
}

class _PressureValues {
  const _PressureValues({required this.systolic, required this.diastolic});

  final int systolic;
  final int diastolic;
}
