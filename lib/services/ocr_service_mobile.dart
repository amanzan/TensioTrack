import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:google_generative_ai/google_generative_ai.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:http/http.dart' as http;

import 'ocr_service.dart';

/// Retorna la instancia de servicio OCR móvil.
OcrService getOcrService() => GeminiOcrService();

class GeminiOcrService implements OcrService {
  /// Clave API de Gemini.
  static const _geminiApiKey = String.fromEnvironment('GEMINI_API_KEY');

  /// Clave API de Groq.
  static const _groqApiKey = String.fromEnvironment('GROQ_API_KEY');

  /// Timeout máximo para las peticiones a proveedores cloud.
  static const _timeout = Duration(seconds: 60);
  static const _maxCloudAttempts = 4;

  /// FLAG TEMPORAL PARA PRUEBAS
  /// Si es true, la app utiliza prioritariamente el lector LCD local/offline.
  /// Si es false, usa Gemini en la nube como primera opción y OCR local como respaldo.
  static bool forceOfflineOcr = false;

  /// FLAG TEMPORAL PARA PRUEBAS
  /// Si es true, omite Gemini y realiza los intentos cloud solo con Groq.
  static bool forceGroqOcr = const bool.fromEnvironment('FORCE_GROQ_OCR');

  static DateTime? _geminiQuotaBlockedUntil;

  @override
  Future<OcrResult?> recognizePressure(
    String imagePath,
    Uint8List imageBytes,
  ) async {
    if (forceOfflineOcr) {
      debugPrint(
        'TensioTrack: forceOfflineOcr activo. Ejecutando OCR LCD offline...',
      );
      try {
        final sevenSegmentResult = await _recognizeWithSevenSegment(imageBytes);
        if (sevenSegmentResult != null) return sevenSegmentResult;
      } catch (e) {
        debugPrint('TensioTrack: Falló el OCR LCD offline ($e).');
      }

      try {
        debugPrint('TensioTrack: Reintentando offline con ML Kit local...');
        return await _recognizeWithMlKit(imagePath, imageBytes);
      } catch (e) {
        debugPrint('TensioTrack: Falló también ML Kit en modo offline ($e).');
        return null;
      }
    }

    try {
      debugPrint(
        forceGroqOcr
            ? 'TensioTrack: Modo cloud forzado a solo Groq.'
            : 'TensioTrack: Modo cloud alterno Gemini/Groq.',
      );
      final cloudResult = await _recognizeWithCloudProvidersWithRetries(
        imageBytes,
      );
      if (cloudResult != null) return cloudResult;

      debugPrint(
        'TensioTrack: Ningún proveedor cloud devolvió lectura tras $_maxCloudAttempts intentos. Usando fallback OCR LCD offline...',
      );
      final sevenSegmentResult = await _recognizeWithSevenSegment(imageBytes);
      if (sevenSegmentResult != null) return sevenSegmentResult;

      debugPrint(
        'TensioTrack: OCR LCD devolvió nulo. Usando fallback offline con ML Kit...',
      );
      return await _recognizeWithMlKit(imagePath, imageBytes);
    } on TimeoutException {
      debugPrint(
        'TensioTrack Cloud TIMEOUT tras $_maxCloudAttempts intentos: Conmutando a OCR LCD offline...',
      );
      final sevenSegmentResult = await _recognizeWithSevenSegment(imageBytes);
      return sevenSegmentResult ??
          await _recognizeWithMlKit(imagePath, imageBytes);
    } catch (e) {
      debugPrint(
        'TensioTrack Cloud ERROR tras $_maxCloudAttempts intentos ($e): Conmutando a OCR LCD offline...',
      );
      final sevenSegmentResult = await _recognizeWithSevenSegment(imageBytes);
      return sevenSegmentResult ??
          await _recognizeWithMlKit(imagePath, imageBytes);
    }
  }

  Future<OcrResult?> _recognizeWithCloudProvidersWithRetries(
    Uint8List imageBytes,
  ) async {
    Object? lastError;

    for (var attempt = 1; attempt <= _maxCloudAttempts; attempt++) {
      final provider = _cloudProviderForAttempt(attempt);

      try {
        debugPrint(
          'TensioTrack ${provider.label}: intento $attempt/$_maxCloudAttempts...',
        );
        final result = switch (provider) {
          _CloudOcrProvider.gemini => await _recognizeWithGemini(imageBytes),
          _CloudOcrProvider.groq => await _recognizeWithGroq(imageBytes),
        };
        if (result != null) return result;
      } catch (e) {
        lastError = e;
        if (provider == _CloudOcrProvider.gemini && _isGeminiQuotaError(e)) {
          _rememberGeminiQuotaCooldown(e);
        }
        debugPrint(
          'TensioTrack ${provider.label}: intento $attempt/$_maxCloudAttempts falló ($e).',
        );
      }

      if (attempt < _maxCloudAttempts) {
        await Future<void>.delayed(Duration(milliseconds: 700 * attempt));
      }
    }

    if (lastError != null) throw lastError;
    return null;
  }

  _CloudOcrProvider _cloudProviderForAttempt(int attempt) {
    if (forceGroqOcr) return _CloudOcrProvider.groq;

    final preferred = attempt.isOdd
        ? _CloudOcrProvider.gemini
        : _CloudOcrProvider.groq;
    if (preferred == _CloudOcrProvider.gemini && _isGeminiQuotaBlocked()) {
      debugPrint(
        'TensioTrack Gemini: cuota agotada temporalmente. Usando Groq en este intento.',
      );
      return _CloudOcrProvider.groq;
    }

    return preferred;
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

  /// OCR local especializado para pantallas LCD de 7 segmentos.
  Future<OcrResult?> _recognizeWithSevenSegment(Uint8List imageBytes) async {
    final image = await _decodeGrayImage(imageBytes);
    final candidates = <_PressureRead>[];
    var layoutIndex = 0;

    for (final layout in _sevenSegmentLayoutsFor(image)) {
      layoutIndex++;
      final sys = _readSevenSegmentNumber(
        image,
        layout.systolicRect,
        allowedDigits: const [3, 2],
        minValue: 70,
        maxValue: 250,
      );
      final dia = _readSevenSegmentNumber(
        image,
        layout.diastolicRect,
        allowedDigits: const [2, 3],
        minValue: 40,
        maxValue: 150,
      );

      debugPrint(
        'TensioTrack LCD OCR layout $layoutIndex '
        '(${layout.detectedFromImage ? 'detectado' : 'default'}): '
        'SYS=${sys?.value}/${sys?.digits}/${sys?.confidence.toStringAsFixed(2)}, '
        'DIA=${dia?.value}/${dia?.digits}/${dia?.confidence.toStringAsFixed(2)}',
      );

      if (sys == null || dia == null || sys.value <= dia.value) {
        continue;
      }

      final pressureGap = (sys.value - dia.value).clamp(0, 90) / 90.0;
      final layoutBonus = layout.detectedFromImage ? 0.06 : 0.0;
      final confidence = ((sys.confidence + dia.confidence) / 2 + layoutBonus)
          .clamp(0.0, 0.98);

      candidates.add(
        _PressureRead(
          systolic: sys,
          diastolic: dia,
          confidence: (confidence * (0.88 + pressureGap * 0.12)).clamp(
            0.0,
            0.98,
          ),
        ),
      );
    }

    if (candidates.isEmpty) {
      debugPrint('TensioTrack LCD OCR: no se encontraron candidatos válidos.');
      return null;
    }

    candidates.sort((a, b) => b.confidence.compareTo(a.confidence));
    final best = candidates.first;
    if (best.confidence < 0.54) {
      debugPrint(
        'TensioTrack LCD OCR: mejor candidato descartado por baja confianza '
        'SYS=${best.systolic.value}, DIA=${best.diastolic.value}, '
        'conf=${best.confidence.toStringAsFixed(2)}',
      );
      return null;
    }

    debugPrint(
      'TensioTrack LCD OCR RESULTADO: '
      'SYS=${best.systolic.value} (${best.systolic.digits}), '
      'DIA=${best.diastolic.value} (${best.diastolic.digits}), '
      'conf=${best.confidence.toStringAsFixed(2)}',
    );

    return OcrResult(
      systolic: best.systolic.value,
      diastolic: best.diastolic.value,
      systolicBox: best.systolic.imageRect,
      diastolicBox: best.diastolic.imageRect,
      imageWidth: image.width.toDouble(),
      imageHeight: image.height.toDouble(),
      confidence: best.confidence,
      engineName: 'LCD 7 segmentos (Offline)',
    );
  }

  Future<_GrayImage> _decodeGrayImage(Uint8List imageBytes) async {
    final codec = await instantiateImageCodec(imageBytes);
    final frame = await codec.getNextFrame();
    final image = frame.image;
    final byteData = await image.toByteData(format: ImageByteFormat.rawRgba);
    if (byteData == null) {
      throw StateError('No se pudo leer la imagen en formato RGBA.');
    }

    final rgba = byteData.buffer.asUint8List();
    final grays = Uint8List(image.width * image.height);
    for (var i = 0; i < rgba.length; i += 4) {
      final r = rgba[i];
      final g = rgba[i + 1];
      final b = rgba[i + 2];
      grays[i ~/ 4] = (0.299 * r + 0.587 * g + 0.114 * b).round();
    }

    return _GrayImage(image.width, image.height, grays);
  }

  List<_SevenSegmentLayout> _sevenSegmentLayoutsFor(_GrayImage image) {
    final w = image.width.toDouble();
    final h = image.height.toDouble();
    final layouts = <_SevenSegmentLayout>[];
    final displayRect = _detectDisplayRect(image);

    if (displayRect != null) {
      final dw = displayRect.width;
      final dh = displayRect.height;
      layouts.addAll([
        _SevenSegmentLayout(
          systolicRect: Rect.fromLTWH(
            displayRect.left + dw * 0.16,
            displayRect.top + dh * 0.06,
            dw * 0.68,
            dh * 0.33,
          ),
          diastolicRect: Rect.fromLTWH(
            displayRect.left + dw * 0.22,
            displayRect.top + dh * 0.36,
            dw * 0.62,
            dh * 0.30,
          ),
          detectedFromImage: true,
        ),
        _SevenSegmentLayout(
          systolicRect: Rect.fromLTWH(
            displayRect.left + dw * 0.18,
            displayRect.top + dh * 0.08,
            dw * 0.64,
            dh * 0.29,
          ),
          diastolicRect: Rect.fromLTWH(
            displayRect.left + dw * 0.25,
            displayRect.top + dh * 0.38,
            dw * 0.57,
            dh * 0.26,
          ),
          detectedFromImage: true,
        ),
      ]);
      debugPrint('TensioTrack LCD OCR: pantalla detectada $displayRect');
      return layouts;
    }

    layouts.addAll([
      _SevenSegmentLayout(
        systolicRect: Rect.fromLTWH(w * 0.30, h * 0.18, w * 0.52, h * 0.23),
        diastolicRect: Rect.fromLTWH(w * 0.30, h * 0.43, w * 0.52, h * 0.23),
      ),
      _SevenSegmentLayout(
        systolicRect: Rect.fromLTWH(w * 0.24, h * 0.16, w * 0.62, h * 0.25),
        diastolicRect: Rect.fromLTWH(w * 0.24, h * 0.42, w * 0.62, h * 0.25),
      ),
      _SevenSegmentLayout(
        systolicRect: Rect.fromLTWH(w * 0.34, h * 0.20, w * 0.48, h * 0.21),
        diastolicRect: Rect.fromLTWH(w * 0.34, h * 0.45, w * 0.48, h * 0.21),
      ),
    ]);

    layouts.addAll(_detectSevenSegmentRows(image));
    return layouts;
  }

  Rect? _detectDisplayRect(_GrayImage image) {
    final scanRect = _clipRect(
      Rect.fromLTWH(
        image.width * 0.22,
        image.height * 0.12,
        image.width * 0.62,
        image.height * 0.55,
      ),
      image.width,
      image.height,
    );

    final left = scanRect.left.round();
    final top = scanRect.top.round();
    final right = scanRect.right.round();
    final bottom = scanRect.bottom.round();
    final components = <_ImageComponent>[];

    for (final threshold in const [95, 105, 115, 125, 135]) {
      final visited = Uint8List(image.width * image.height);

      for (var y = top; y < bottom; y++) {
        for (var x = left; x < right; x++) {
          final index = y * image.width + x;
          if (visited[index] == 1 || image.grayAt(x, y) > threshold) {
            continue;
          }

          final stack = <int>[index];
          visited[index] = 1;
          var area = 0;
          var minX = x;
          var maxX = x;
          var minY = y;
          var maxY = y;

          while (stack.isNotEmpty) {
            final current = stack.removeLast();
            area++;
            final cx = current % image.width;
            final cy = current ~/ image.width;
            minX = math.min(minX, cx);
            maxX = math.max(maxX, cx);
            minY = math.min(minY, cy);
            maxY = math.max(maxY, cy);

            void visit(int nx, int ny) {
              if (nx < left || nx >= right || ny < top || ny >= bottom) return;
              final ni = ny * image.width + nx;
              if (visited[ni] == 1 || image.grayAt(nx, ny) > threshold) {
                return;
              }
              visited[ni] = 1;
              stack.add(ni);
            }

            visit(cx - 1, cy);
            visit(cx + 1, cy);
            visit(cx, cy - 1);
            visit(cx, cy + 1);
          }

          final rect = Rect.fromLTRB(
            minX.toDouble(),
            minY.toDouble(),
            (maxX + 1).toDouble(),
            (maxY + 1).toDouble(),
          );
          final aspect = rect.width / rect.height;
          final fillRatio = area / (rect.width * rect.height);
          final validSize =
              rect.width > image.width * 0.25 &&
              rect.width < image.width * 0.55 &&
              rect.height > image.height * 0.22 &&
              rect.height < image.height * 0.50;
          if (!validSize ||
              aspect < 0.55 ||
              aspect > 1.05 ||
              fillRatio < 0.045) {
            continue;
          }

          components.add(_ImageComponent(rect: rect, area: area));
        }
      }
    }

    if (components.isEmpty) return null;
    components.sort((a, b) => b.area.compareTo(a.area));
    return components.first.rect;
  }

  List<_SevenSegmentLayout> _detectSevenSegmentRows(_GrayImage image) {
    final scanRect = Rect.fromLTWH(
      image.width * 0.22,
      image.height * 0.10,
      image.width * 0.68,
      image.height * 0.68,
    );
    final region = _buildBinaryRegion(image, scanRect);
    final projection = List<int>.filled(region.height, 0);
    for (var y = 0; y < region.height; y++) {
      var count = 0;
      for (var x = 0; x < region.width; x++) {
        if (region.isForeground(x, y)) count++;
      }
      projection[y] = count;
    }

    final threshold = math.max(4, (region.width * 0.018).round());
    final minHeight = math.max(12, (region.height * 0.07).round());
    final clusters = <_Range>[];
    var start = -1;

    for (var y = 0; y < projection.length; y++) {
      if (projection[y] >= threshold) {
        start = start == -1 ? y : start;
      } else if (start != -1) {
        if (y - start >= minHeight) clusters.add(_Range(start, y - 1));
        start = -1;
      }
    }
    if (start != -1 && projection.length - start >= minHeight) {
      clusters.add(_Range(start, projection.length - 1));
    }

    if (clusters.length < 2) return const [];

    clusters.sort((a, b) {
      final scoreA = _rowClusterScore(projection, a);
      final scoreB = _rowClusterScore(projection, b);
      return scoreB.compareTo(scoreA);
    });

    final selected = <_Range>[];
    for (final cluster in clusters) {
      if (selected.every(
        (other) => (cluster.center - other.center).abs() > minHeight,
      )) {
        selected.add(cluster);
      }
      if (selected.length == 2) break;
    }
    if (selected.length < 2) return const [];

    selected.sort((a, b) => a.start.compareTo(b.start));
    final padY = image.height * 0.018;
    final x = scanRect.left + scanRect.width * 0.08;
    final width = scanRect.width * 0.82;

    Rect rectFor(_Range range) {
      final top = scanRect.top + range.start - padY;
      final height = range.length + padY * 2;
      return Rect.fromLTWH(x, top, width, height);
    }

    return [
      _SevenSegmentLayout(
        systolicRect: rectFor(selected[0]),
        diastolicRect: rectFor(selected[1]),
        detectedFromImage: true,
      ),
    ];
  }

  int _rowClusterScore(List<int> projection, _Range range) {
    var score = 0;
    for (var y = range.start; y <= range.end; y++) {
      score += projection[y];
    }
    return score;
  }

  _SevenSegmentRead? _readSevenSegmentNumber(
    _GrayImage image,
    Rect rect, {
    required List<int> allowedDigits,
    required int minValue,
    required int maxValue,
  }) {
    final region = _buildBinaryRegion(image, rect);
    final bounds = region.foregroundBounds();
    if (bounds == null) return null;

    final padded = bounds
        .inflate(3)
        .intersect(
          Rect.fromLTWH(
            0,
            0,
            region.width.toDouble(),
            region.height.toDouble(),
          ),
        );

    final reads = <_SevenSegmentRead>[];
    for (final digitCount in allowedDigits) {
      final read = _readFixedSevenSegmentNumber(region, padded, digitCount);
      if (read == null || read.value < minValue || read.value > maxValue) {
        continue;
      }
      reads.add(read);
    }

    if (reads.isEmpty) return null;

    final preferredDigitCount = allowedDigits.first;
    reads.sort((a, b) {
      final aScore =
          a.confidence + (a.digitCount == preferredDigitCount ? 0.11 : 0);
      final bScore =
          b.confidence + (b.digitCount == preferredDigitCount ? 0.11 : 0);
      return bScore.compareTo(aScore);
    });
    return reads.first;
  }

  _SevenSegmentRead? _readFixedSevenSegmentNumber(
    _BinaryRegion region,
    Rect localBounds,
    int digitCount,
  ) {
    final digits = <int>[];
    var confidenceSum = 0.0;
    final cells = _digitCellsFromProjection(region, localBounds, digitCount);

    for (var i = 0; i < digitCount; i++) {
      final cell = cells[i];

      final digit = _classifySevenSegmentDigit(region, cell);
      if (digit == null) return null;

      digits.add(digit.value);
      confidenceSum += digit.confidence;
    }

    if (digits.isEmpty || digits.first == 0 && digitCount == 3) {
      return null;
    }

    final value = int.tryParse(digits.join());
    if (value == null) return null;

    final imageRect = region.toImageRect(localBounds);
    return _SevenSegmentRead(
      value: value,
      digits: digits.join(),
      digitCount: digitCount,
      imageRect: imageRect,
      confidence: (confidenceSum / digits.length).clamp(0.0, 1.0),
    );
  }

  List<Rect> _digitCellsFromProjection(
    _BinaryRegion region,
    Rect localBounds,
    int digitCount,
  ) {
    final left = localBounds.left.floor().clamp(0, region.width - 1);
    final right = localBounds.right.ceil().clamp(left + 1, region.width);
    final top = localBounds.top.floor().clamp(0, region.height - 1);
    final bottom = localBounds.bottom.ceil().clamp(top + 1, region.height);
    final minColumnCount = math.max(3, ((bottom - top) * 0.08).round());
    final clusters = <_ColumnCluster>[];

    var start = -1;
    var maxCount = 0;
    var area = 0;

    void closeCluster(int endExclusive) {
      if (start == -1) return;
      final width = endExclusive - start;
      if (width >= 3 && maxCount >= minColumnCount) {
        clusters.add(
          _ColumnCluster(
            start: start,
            end: endExclusive,
            maxCount: maxCount,
            area: area,
          ),
        );
      }
      start = -1;
      maxCount = 0;
      area = 0;
    }

    for (var x = left; x < right; x++) {
      var count = 0;
      for (var y = top; y < bottom; y++) {
        if (region.isForeground(x, y)) count++;
      }

      if (count >= minColumnCount) {
        start = start == -1 ? x : start;
        maxCount = math.max(maxCount, count);
        area += count;
      } else {
        closeCluster(x);
      }
    }
    closeCluster(right);

    clusters.sort((a, b) => b.area.compareTo(a.area));
    final selected = clusters.take(digitCount).toList()
      ..sort((a, b) => a.start.compareTo(b.start));

    if (selected.length < digitCount) {
      final cellWidth = localBounds.width / digitCount;
      return [
        for (var i = 0; i < digitCount; i++)
          Rect.fromLTWH(
            localBounds.left + cellWidth * i,
            localBounds.top,
            cellWidth,
            localBounds.height,
          ).deflate(math.max(1, cellWidth * 0.04)),
      ];
    }

    final wideWidths = selected
        .map((cluster) => cluster.width)
        .where((width) => width >= localBounds.width / (digitCount * 2.4))
        .toList();
    final expectedDigitWidth = wideWidths.isEmpty
        ? localBounds.width / digitCount
        : wideWidths.reduce(math.max).toDouble();

    return [
      for (final cluster in selected)
        _cellForColumnCluster(
          cluster,
          localBounds,
          expectedDigitWidth,
          region.width,
        ),
    ];
  }

  Rect _cellForColumnCluster(
    _ColumnCluster cluster,
    Rect localBounds,
    double expectedDigitWidth,
    int regionWidth,
  ) {
    final clusterWidth = cluster.width.toDouble();
    final heightPad = math.max(1.0, localBounds.height * 0.02);
    late double left;
    late double width;

    if (clusterWidth < expectedDigitWidth * 0.45) {
      width = clusterWidth + expectedDigitWidth * 0.14;
      final right = cluster.end + expectedDigitWidth * 0.07;
      left = right - width;
    } else {
      width = clusterWidth + expectedDigitWidth * 0.14;
      left = cluster.start - expectedDigitWidth * 0.07;
    }

    left = left.clamp(localBounds.left, regionWidth - width);
    return Rect.fromLTWH(
      left,
      localBounds.top + heightPad,
      width,
      localBounds.height - heightPad * 2,
    );
  }

  _DigitRead? _classifySevenSegmentDigit(_BinaryRegion region, Rect cell) {
    if (cell.width < cell.height * 0.35) {
      final verticalDensity = region.densityIn(
        cell,
        const Rect.fromLTWH(0.18, 0.10, 0.64, 0.80),
      );
      if (verticalDensity > 0.14) {
        return const _DigitRead(1, 0.96);
      }
    }

    final densities = <double>[
      region.densityIn(cell, const Rect.fromLTWH(0.16, 0.03, 0.68, 0.20)),
      region.densityIn(cell, const Rect.fromLTWH(0.68, 0.15, 0.24, 0.34)),
      region.densityIn(cell, const Rect.fromLTWH(0.68, 0.51, 0.24, 0.34)),
      region.densityIn(cell, const Rect.fromLTWH(0.16, 0.78, 0.68, 0.19)),
      region.densityIn(cell, const Rect.fromLTWH(0.08, 0.51, 0.24, 0.34)),
      region.densityIn(cell, const Rect.fromLTWH(0.08, 0.15, 0.24, 0.34)),
      region.densityIn(cell, const Rect.fromLTWH(0.18, 0.40, 0.64, 0.18)),
    ];

    final maxDensity = densities.fold<double>(0, math.max);
    if (maxDensity < 0.035) return null;

    final patterns = <int, List<int>>{
      0: const [0, 1, 2, 3, 4, 5],
      1: const [1, 2],
      2: const [0, 1, 3, 4, 6],
      3: const [0, 1, 2, 3, 6],
      4: const [1, 2, 5, 6],
      5: const [0, 2, 3, 5, 6],
      6: const [0, 2, 3, 4, 5, 6],
      7: const [0, 1, 2],
      8: const [0, 1, 2, 3, 4, 5, 6],
      9: const [0, 1, 2, 3, 5, 6],
    };

    var bestDigit = -1;
    var bestDistance = double.infinity;
    var secondDistance = double.infinity;

    for (final entry in patterns.entries) {
      final activeSegments = entry.value.toSet();
      var distance = 0.0;

      for (var i = 0; i < densities.length; i++) {
        final normalized = (densities[i] / maxDensity).clamp(0.0, 1.0);
        if (activeSegments.contains(i)) {
          distance += 1.0 - normalized;
        } else {
          distance += normalized * 0.82;
        }
      }

      if (distance < bestDistance) {
        secondDistance = bestDistance;
        bestDistance = distance;
        bestDigit = entry.key;
      } else if (distance < secondDistance) {
        secondDistance = distance;
      }
    }

    if (bestDigit == -1) return null;

    final normalized = [
      for (final density in densities) (density / maxDensity).clamp(0.0, 1.0),
    ];
    if (bestDigit == 4 &&
        normalized[0] > 0.25 &&
        normalized[4] < 0.28 &&
        normalized[6] < 0.24) {
      bestDigit = 7;
      bestDistance = math.min(bestDistance, 1.75);
    }

    final rawConfidence = 1.0 - (bestDistance / 7.0);
    final separation = ((secondDistance - bestDistance) / 3.5).clamp(0.0, 0.22);
    final confidence = (rawConfidence + separation).clamp(0.0, 1.0);
    if (confidence < 0.42) return null;

    return _DigitRead(bestDigit, confidence);
  }

  _BinaryRegion _buildBinaryRegion(_GrayImage image, Rect rect) {
    final clipped = _clipRect(rect, image.width, image.height);
    final width = clipped.width.round().clamp(1, image.width);
    final height = clipped.height.round().clamp(1, image.height);
    final left = clipped.left.round();
    final top = clipped.top.round();
    final grays = Uint8List(width * height);

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        grays[y * width + x] = image.grayAt(left + x, top + y);
      }
    }

    final threshold = _localOtsuThreshold(grays);
    final foreground = Uint8List(width * height);
    for (var i = 0; i < grays.length; i++) {
      foreground[i] = grays[i] <= threshold ? 1 : 0;
    }

    final cleaned = _removeSmallComponents(foreground, width, height);
    return _BinaryRegion(
      width: width,
      height: height,
      foreground: cleaned,
      imageRect: clipped,
    );
  }

  Rect _clipRect(Rect rect, int imageWidth, int imageHeight) {
    final left = rect.left.clamp(0.0, imageWidth - 1.0);
    final top = rect.top.clamp(0.0, imageHeight - 1.0);
    final right = rect.right.clamp(left + 1.0, imageWidth.toDouble());
    final bottom = rect.bottom.clamp(top + 1.0, imageHeight.toDouble());
    return Rect.fromLTRB(left, top, right, bottom);
  }

  int _localOtsuThreshold(Uint8List grays) {
    final histogram = List<int>.filled(256, 0);
    for (final gray in grays) {
      histogram[gray]++;
    }

    var total = grays.length;
    var sum = 0.0;
    for (var i = 0; i < 256; i++) {
      sum += i * histogram[i];
    }

    var sumB = 0.0;
    var weightB = 0;
    var bestVariance = -1.0;
    var threshold = 127;

    for (var t = 0; t < 256; t++) {
      weightB += histogram[t];
      if (weightB == 0) continue;

      final weightF = total - weightB;
      if (weightF == 0) break;

      sumB += t * histogram[t];
      final meanB = sumB / weightB;
      final meanF = (sum - sumB) / weightF;
      final variance =
          weightB.toDouble() * weightF.toDouble() * math.pow(meanB - meanF, 2);

      if (variance > bestVariance) {
        bestVariance = variance.toDouble();
        threshold = t;
      }
    }

    final sorted = Uint8List.fromList(grays)..sort();
    final p10 = sorted[(sorted.length * 0.10).floor()];
    final p35 = sorted[(sorted.length * 0.35).floor()];
    final p55 = sorted[(sorted.length * 0.55).floor()];

    // En LCDs fotografiados, los segmentos apagados suelen quedar como trazos
    // grisáceos. Usar el umbral de Otsu "tal cual" tiende a activar esos trazos
    // fantasma y convierte muchos dígitos en 8. Este umbral se queda en el
    // tercio oscuro de la distribución para conservar solo segmentos encendidos.
    final darkSegmentThreshold = (p10 + (p35 - p10) * 0.55).round();
    final conservativeOtsu = math.min(threshold, p55);
    return math.min(conservativeOtsu, darkSegmentThreshold).clamp(25, 185);
  }

  Uint8List _removeSmallComponents(
    Uint8List foreground,
    int width,
    int height,
  ) {
    final visited = Uint8List(foreground.length);
    final cleaned = Uint8List(foreground.length);
    final minArea = math.max(12, (width * height * 0.0018).round());
    final minLongSide = math.max(5, (math.min(width, height) * 0.055).round());

    for (var i = 0; i < foreground.length; i++) {
      if (foreground[i] == 0 || visited[i] == 1) continue;

      final stack = <int>[i];
      final component = <int>[];
      visited[i] = 1;
      var minX = width;
      var maxX = 0;
      var minY = height;
      var maxY = 0;

      while (stack.isNotEmpty) {
        final current = stack.removeLast();
        component.add(current);
        final x = current % width;
        final y = current ~/ width;
        minX = math.min(minX, x);
        maxX = math.max(maxX, x);
        minY = math.min(minY, y);
        maxY = math.max(maxY, y);

        void visit(int nx, int ny) {
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) return;
          final ni = ny * width + nx;
          if (foreground[ni] == 0 || visited[ni] == 1) return;
          visited[ni] = 1;
          stack.add(ni);
        }

        visit(x - 1, y);
        visit(x + 1, y);
        visit(x, y - 1);
        visit(x, y + 1);
      }

      final componentWidth = maxX - minX + 1;
      final componentHeight = maxY - minY + 1;
      final keep =
          component.length >= minArea ||
          math.max(componentWidth, componentHeight) >= minLongSide;

      if (keep) {
        for (final pixel in component) {
          cleaned[pixel] = 1;
        }
      }
    }

    return cleaned;
  }

  /// OCR en local y offline usando Google ML Kit con preprocesamiento avanzado
  Future<OcrResult?> _recognizeWithMlKit(
    String imagePath,
    Uint8List imageBytes,
  ) async {
    debugPrint(
      'TensioTrack: Inicializando TextRecognizer local de ML Kit con 2 pases...',
    );

    // Decodificar imagen para obtener dimensiones reales y objeto de imagen para recortar
    double imageWidth = 1000;
    double imageHeight = 1000;
    Image? originalImage;
    try {
      final codec = await instantiateImageCodec(imageBytes);
      final frame = await codec.getNextFrame();
      originalImage = frame.image;
      imageWidth = originalImage.width.toDouble();
      imageHeight = originalImage.height.toDouble();
    } catch (e) {
      debugPrint('TensioTrack ERROR: No se pudo decodificar la imagen: $e');
    }

    final textRecognizer = TextRecognizer(script: TextRecognitionScript.latin);
    try {
      // Pase 1: Reconocimiento de texto en la imagen original para buscar etiquetas
      final inputImage = InputImage.fromFilePath(imagePath);
      final recognizedText = await textRecognizer.processImage(inputImage);

      debugPrint('================ PASO 1: OCR DE ALINEACION ================');
      debugPrint('Texto total reconocido:\n"${recognizedText.text}"');

      Rect? sysLabelRect;
      Rect? diaLabelRect;

      for (final block in recognizedText.blocks) {
        for (final line in block.lines) {
          final text = line.text.toUpperCase();
          debugPrint('Alineación - Línea: "$text" | Rect: ${line.boundingBox}');
          if (text.contains('SYS')) {
            sysLabelRect = line.boundingBox;
          } else if (text.contains('DIA')) {
            diaLabelRect = line.boundingBox;
          }
        }
      }
      debugPrint('===========================================================');

      // Calcular regiones de interés (ROI)
      Rect? sysCropRect;
      Rect? diaCropRect;

      if (sysLabelRect != null) {
        // Desplazado horizontalmente más a la derecha (factor 2.2) para saltar el marco negro de plástico de la pantalla LCD
        final labelHeight = sysLabelRect.height;
        final left = sysLabelRect.right + labelHeight * 2.2;
        final top = sysLabelRect.top - labelHeight * 1.0;
        final width = sysLabelRect.width * 6.0;
        final height = labelHeight * 4.5;

        sysCropRect = Rect.fromLTWH(
          left.clamp(0.0, imageWidth - 10),
          top.clamp(0.0, imageHeight - 10),
          width.clamp(10.0, imageWidth - left),
          height.clamp(10.0, imageHeight - top),
        );
        debugPrint('SYS ROI calculada: $sysCropRect');
      }

      if (diaLabelRect != null) {
        final labelHeight = diaLabelRect.height;
        final left = diaLabelRect.right + labelHeight * 2.2;
        final top = diaLabelRect.top - labelHeight * 1.0;
        final width = diaLabelRect.width * 6.0;
        final height = labelHeight * 4.5;

        diaCropRect = Rect.fromLTWH(
          left.clamp(0.0, imageWidth - 10),
          top.clamp(0.0, imageHeight - 10),
          width.clamp(10.0, imageWidth - left),
          height.clamp(10.0, imageHeight - top),
        );
        debugPrint('DIA ROI calculada: $diaCropRect');
      }

      // Si no se encuentran etiquetas, usar proporciones centrales predeterminadas (Omron)
      if (sysCropRect == null && originalImage != null) {
        sysCropRect = Rect.fromLTWH(
          imageWidth * 0.30,
          imageHeight * 0.20,
          imageWidth * 0.50,
          imageHeight * 0.25,
        );
        debugPrint('SYS ROI default: $sysCropRect');
      }
      if (diaCropRect == null && originalImage != null) {
        diaCropRect = Rect.fromLTWH(
          imageWidth * 0.30,
          imageHeight * 0.45,
          imageWidth * 0.50,
          imageHeight * 0.25,
        );
        debugPrint('DIA ROI default: $diaCropRect');
      }

      int sysVal = 0;
      int diaVal = 0;

      final tempPaths = <String>[];

      // Procesar SYS
      if (sysCropRect != null && originalImage != null) {
        try {
          final sysCrop = await _cropImage(originalImage, sysCropRect);

          // 1. Intentar con el recorte crudo (solo escalado)
          final sysRawPath = await _saveImageToTempFile(sysCrop, 'sys_raw');
          tempPaths.add(sysRawPath);
          final sysRawRecText = await textRecognizer.processImage(
            InputImage.fromFilePath(sysRawPath),
          );
          debugPrint('OCR SYS crudo recortado: "${sysRawRecText.text}"');
          sysVal = _extractCleanNumber(sysRawRecText.text) ?? 0;

          // 2. Si falló, aplicar preprocesamiento avanzado
          if (sysVal == 0) {
            final sysBytes = await _preprocessRegion(sysCrop);
            if (sysBytes != null) {
              final path = await _saveRgbaBytesToTempFile(
                sysBytes,
                sysCrop.width,
                sysCrop.height,
                'sys_bin',
              );
              tempPaths.add(path);

              final sysRecText = await textRecognizer.processImage(
                InputImage.fromFilePath(path),
              );
              debugPrint(
                'OCR SYS binarizado adaptativo recortado: "${sysRecText.text}"',
              );
              sysVal = _extractCleanNumber(sysRecText.text) ?? 0;
            }
          }
        } catch (e) {
          debugPrint('Error procesando ROI SYS: $e');
        }
      }

      // Procesar DIA
      if (diaCropRect != null && originalImage != null) {
        try {
          final diaCrop = await _cropImage(originalImage, diaCropRect);

          // 1. Intentar con el recorte crudo (solo escalado)
          final diaRawPath = await _saveImageToTempFile(diaCrop, 'dia_raw');
          tempPaths.add(diaRawPath);
          final diaRawRecText = await textRecognizer.processImage(
            InputImage.fromFilePath(diaRawPath),
          );
          debugPrint('OCR DIA crudo recortado: "${diaRawRecText.text}"');
          diaVal = _extractCleanNumber(diaRawRecText.text) ?? 0;

          // 2. Si falló, aplicar preprocesamiento avanzado
          if (diaVal == 0) {
            final diaBytes = await _preprocessRegion(diaCrop);
            if (diaBytes != null) {
              final path = await _saveRgbaBytesToTempFile(
                diaBytes,
                diaCrop.width,
                diaCrop.height,
                'dia_bin',
              );
              tempPaths.add(path);

              final diaRecText = await textRecognizer.processImage(
                InputImage.fromFilePath(path),
              );
              debugPrint(
                'OCR DIA binarizado adaptativo recortado: "${diaRecText.text}"',
              );
              diaVal = _extractCleanNumber(diaRecText.text) ?? 0;
            }
          }
        } catch (e) {
          debugPrint('Error procesando ROI DIA: $e');
        }
      }

      // Limpiar archivos temporales (comentado temporalmente para permitir inspección/depuración)
      /*
      for (final p in tempPaths) {
        try {
          final f = File(p);
          if (await f.exists()) {
            await f.delete();
          }
        } catch (_) {}
      }
      */

      // Si falló el reconocimiento segmentado por completo, aplicar el fallback heurístico en la imagen original
      if (sysVal == 0 || diaVal == 0) {
        debugPrint(
          'Advertencia: El OCR segmentado falló. Aplicando fallback heurístico en imagen original...',
        );
        final List<MapEntry<int, Rect>> candidateNumbers = [];
        for (final block in recognizedText.blocks) {
          for (final line in block.lines) {
            final matches = RegExp(r'\d+').allMatches(line.text);
            for (final match in matches) {
              final val = int.tryParse(match.group(0)!);
              if (val != null && val >= 35 && val <= 240) {
                candidateNumbers.add(MapEntry(val, line.boundingBox));
              }
            }
          }
        }

        if (candidateNumbers.length >= 2) {
          candidateNumbers.sort((a, b) => a.value.top.compareTo(b.value.top));
          sysVal = candidateNumbers[0].key;
          diaVal = candidateNumbers[1].key;
        } else if (candidateNumbers.length == 1) {
          final val = candidateNumbers[0].key;
          if (val >= 95) {
            sysVal = val;
          } else {
            diaVal = val;
          }
        }
      }

      if (sysVal == 0 && diaVal == 0) {
        debugPrint('TensioTrack: ML Kit local no encontró ningún valor.');
        return null;
      }

      // Asegurar que SYS > DIA por lógica fisiológica básica
      if (sysVal > 0 && diaVal > 0 && sysVal < diaVal) {
        final temp = sysVal;
        sysVal = diaVal;
        diaVal = temp;
      }

      debugPrint(
        'TensioTrack: ML Kit local extrajo exitosamente SYS=$sysVal, DIA=$diaVal',
      );

      return OcrResult(
        systolic: sysVal,
        diastolic: diaVal,
        systolicBox: sysCropRect,
        diastolicBox: diaCropRect,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
        confidence: (sysVal > 0 && diaVal > 0) ? 0.90 : 0.50,
        engineName: 'Google ML Kit (Local)',
      );
    } finally {
      await textRecognizer.close();
    }
  }

  /// Recorta un sub-rectángulo de la imagen original, escalándolo por un factor de 2.0
  /// para garantizar que la imagen resultante tenga un tamaño óptimo para el motor de ML Kit.
  Future<Image> _cropImage(Image src, Rect cropRect) async {
    final recorder = PictureRecorder();
    final canvas = Canvas(recorder);
    final srcRect = cropRect;

    // Escalar por 2.0x para mejorar la resolución y que ML Kit reconozca mejor los dígitos pequeños
    final double scale = 2.0;
    final destRect = Rect.fromLTWH(
      0,
      0,
      cropRect.width * scale,
      cropRect.height * scale,
    );

    canvas.drawImageRect(
      src,
      srcRect,
      destRect,
      Paint()..filterQuality = FilterQuality.high,
    );
    final picture = recorder.endRecording();
    return await picture.toImage(
      (cropRect.width * scale).toInt(),
      (cropRect.height * scale).toInt(),
    );
  }

  /// Realiza binarización adaptativa local (Bradley-Roth) y dilatación binaria sobre la región recortada
  Future<Uint8List?> _preprocessRegion(Image croppedImage) async {
    final byteData = await croppedImage.toByteData(
      format: ImageByteFormat.rawRgba,
    );
    if (byteData == null) return null;
    final pixels = byteData.buffer.asUint8List();
    final w = croppedImage.width;
    final h = croppedImage.height;

    // 1. Convertir a escala de grises
    final grays = Uint8List(w * h);
    for (int i = 0; i < pixels.length; i += 4) {
      int r = pixels[i];
      int g = pixels[i + 1];
      int b = pixels[i + 2];
      grays[i ~/ 4] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0, 255);
    }

    // 2. Construir imagen integral
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

    // 3. Aplicar umbral adaptativo local (Bradley-Roth)
    final binarized = Uint8List(pixels.length);
    final int s = (w / 8).round().clamp(7, 100); // Tamaño de ventana local
    const double t =
        0.15; // Un pixel debe ser un 15% más oscuro que su entorno local para ser negro

    int blackCount = 0;
    int whiteCount = 0;

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int x1 = (x - s ~/ 2).clamp(0, w - 1);
        int x2 = (x + s ~/ 2).clamp(0, w - 1);
        int y1 = (y - s ~/ 2).clamp(0, h - 1);
        int y2 = (y + s ~/ 2).clamp(0, h - 1);

        int count = (x2 - x1 + 1) * (y2 - y1 + 1);

        // Calcular la suma de la ventana usando la imagen integral
        int sum = integral[y2 * w + x2];
        if (x1 > 0) {
          sum -= integral[y2 * w + (x1 - 1)];
        }
        if (y1 > 0) {
          sum -= integral[(y1 - 1) * w + x2];
        }
        if (x1 > 0 && y1 > 0) {
          sum += integral[(y1 - 1) * w + (x1 - 1)];
        }

        int gray = grays[y * w + x];

        // Binarizar: si el pixel es significativamente más oscuro que el promedio local, es texto (negro)
        int color = (gray * count < sum * (1.0 - t)) ? 0 : 255;

        int idx = (y * w + x) * 4;
        binarized[idx] = color;
        binarized[idx + 1] = color;
        binarized[idx + 2] = color;
        binarized[idx + 3] = 255; // Alpha

        if (color == 0) {
          blackCount++;
        } else {
          whiteCount++;
        }
      }
    }

    debugPrint(
      'TensioTrack Preprocess [${croppedImage.hashCode}]: Adaptive binarized black pixels = $blackCount (Ratio: ${(blackCount / (w * h) * 100).toStringAsFixed(1)}%), white = $whiteCount',
    );

    // 4. Dilatación binaria doble para rellenar huecos en los 7 segmentos
    final dilated1 = _dilate(binarized, w, h);
    final dilated2 = _dilate(dilated1, w, h);

    int dilatedBlackCount = 0;
    for (int i = 0; i < dilated2.length; i += 4) {
      if (dilated2[i] == 0) {
        dilatedBlackCount++;
      }
    }
    debugPrint(
      'TensioTrack Preprocess [${croppedImage.hashCode}]: Dilated black pixels = $dilatedBlackCount (Ratio: ${(dilatedBlackCount / (w * h) * 100).toStringAsFixed(1)}%)',
    );

    return dilated2;
  }

  /// Aplica una dilatación morfológica 3x3 (expansión de pixeles negros de foreground)
  Uint8List _dilate(Uint8List source, int width, int height) {
    final output = Uint8List(source.length);
    output.fillRange(0, output.length, 255);
    for (int i = 3; i < output.length; i += 4) {
      output[i] = 255;
    }

    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        bool hasForeground = false;
        for (int ky = -1; ky <= 1; ky++) {
          for (int kx = -1; kx <= 1; kx++) {
            int idx = ((y + ky) * width + (x + kx)) * 4;
            if (source[idx] == 0) {
              hasForeground = true;
              break;
            }
          }
          if (hasForeground) break;
        }

        int outIdx = (y * width + x) * 4;
        if (hasForeground) {
          output[outIdx] = 0;
          output[outIdx + 1] = 0;
          output[outIdx + 2] = 0;
        }
      }
    }
    return output;
  }

  /// Guarda una imagen Dart UI en un archivo PNG temporal y retorna la ruta
  Future<String> _saveImageToTempFile(Image image, String prefix) async {
    final byteData = await image.toByteData(format: ImageByteFormat.png);
    final pngBytes = byteData!.buffer.asUint8List();

    final tempDir = Directory.systemTemp;
    final tempFile = File('${tempDir.path}/temp_ocr_crop_$prefix.png');
    await tempFile.writeAsBytes(pngBytes);
    debugPrint(
      'TensioTrack Preprocess [${image.hashCode}]: Saved cropped $prefix image to: ${tempFile.path} (${pngBytes.length} bytes)',
    );
    return tempFile.path;
  }

  /// Guarda los bytes RGBA en un archivo PNG temporal y retorna la ruta
  Future<String> _saveRgbaBytesToTempFile(
    Uint8List pixels,
    int width,
    int height,
    String prefix,
  ) async {
    final completer = Completer<Image>();
    decodeImageFromPixels(
      pixels,
      width,
      height,
      PixelFormat.rgba8888,
      completer.complete,
    );
    final image = await completer.future;
    return await _saveImageToTempFile(image, prefix);
  }

  /// Extrae números de cadenas y aplica mapeo corrector de 7 segmentos
  int? _extractCleanNumber(String rawText) {
    final lines = rawText.split('\n');
    final List<int> candidates = [];

    for (var line in lines) {
      String mapped = line.toUpperCase().trim();

      // Mapeos de caracteres de 7 segmentos a dígitos equivalentes
      mapped = mapped.replaceAll(RegExp(r'[ILi|!\\\\\\\\[\\\\\\\\]]'), '1');
      mapped = mapped.replaceAll(RegExp(r'[OoD]'), '0');
      mapped = mapped.replaceAll(RegExp(r'[S]'), '5');
      mapped = mapped.replaceAll(RegExp(r'[B]'), '8');
      mapped = mapped.replaceAll(RegExp(r'[Z]'), '2');
      mapped = mapped.replaceAll(RegExp(r'[Gg]'), '6');
      mapped = mapped.replaceAll(RegExp(r'[A]'), '4');
      mapped = mapped.replaceAll(RegExp(r'[T]'), '7');

      // Buscar grupos de dígitos en la línea
      final matches = RegExp(r'\d+').allMatches(mapped);
      for (final match in matches) {
        final val = int.tryParse(match.group(0)!);
        if (val != null && val >= 30 && val <= 250) {
          candidates.add(val);
        }
      }
    }

    if (candidates.isEmpty) return null;
    return candidates.first;
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
      model: 'gemini-2.5-flash',
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
    final values = _parseGroqBpResponse(content);
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

  _PressureValues? _parseGroqBpResponse(String content) {
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
  groq('Groq');

  const _CloudOcrProvider(this.label);

  final String label;
}

class _PressureValues {
  const _PressureValues({required this.systolic, required this.diastolic});

  final int systolic;
  final int diastolic;
}

class _GrayImage {
  const _GrayImage(this.width, this.height, this.grays);

  final int width;
  final int height;
  final Uint8List grays;

  int grayAt(int x, int y) => grays[y * width + x];
}

class _BinaryRegion {
  const _BinaryRegion({
    required this.width,
    required this.height,
    required this.foreground,
    required this.imageRect,
  });

  final int width;
  final int height;
  final Uint8List foreground;
  final Rect imageRect;

  bool isForeground(int x, int y) => foreground[y * width + x] == 1;

  Rect? foregroundBounds() {
    var minX = width;
    var minY = height;
    var maxX = -1;
    var maxY = -1;

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        if (!isForeground(x, y)) continue;
        minX = math.min(minX, x);
        minY = math.min(minY, y);
        maxX = math.max(maxX, x);
        maxY = math.max(maxY, y);
      }
    }

    if (maxX == -1 || maxY == -1) return null;
    return Rect.fromLTRB(
      minX.toDouble(),
      minY.toDouble(),
      (maxX + 1).toDouble(),
      (maxY + 1).toDouble(),
    );
  }

  Rect toImageRect(Rect localRect) {
    return Rect.fromLTWH(
      imageRect.left + localRect.left,
      imageRect.top + localRect.top,
      localRect.width,
      localRect.height,
    );
  }

  double densityIn(Rect cell, Rect normalizedZone) {
    final zone = Rect.fromLTWH(
      cell.left + cell.width * normalizedZone.left,
      cell.top + cell.height * normalizedZone.top,
      cell.width * normalizedZone.width,
      cell.height * normalizedZone.height,
    );

    final left = zone.left.floor().clamp(0, width - 1);
    final top = zone.top.floor().clamp(0, height - 1);
    final right = zone.right.ceil().clamp(left + 1, width);
    final bottom = zone.bottom.ceil().clamp(top + 1, height);

    var count = 0;
    var total = 0;
    for (var y = top; y < bottom; y++) {
      for (var x = left; x < right; x++) {
        total++;
        if (isForeground(x, y)) count++;
      }
    }

    if (total == 0) return 0;
    return count / total;
  }
}

class _SevenSegmentLayout {
  const _SevenSegmentLayout({
    required this.systolicRect,
    required this.diastolicRect,
    this.detectedFromImage = false,
  });

  final Rect systolicRect;
  final Rect diastolicRect;
  final bool detectedFromImage;
}

class _SevenSegmentRead {
  const _SevenSegmentRead({
    required this.value,
    required this.digits,
    required this.digitCount,
    required this.imageRect,
    required this.confidence,
  });

  final int value;
  final String digits;
  final int digitCount;
  final Rect imageRect;
  final double confidence;
}

class _PressureRead {
  const _PressureRead({
    required this.systolic,
    required this.diastolic,
    required this.confidence,
  });

  final _SevenSegmentRead systolic;
  final _SevenSegmentRead diastolic;
  final double confidence;
}

class _DigitRead {
  const _DigitRead(this.value, this.confidence);

  final int value;
  final double confidence;
}

class _Range {
  const _Range(this.start, this.end);

  final int start;
  final int end;

  int get length => end - start + 1;
  double get center => (start + end) / 2.0;
}

class _ImageComponent {
  const _ImageComponent({required this.rect, required this.area});

  final Rect rect;
  final int area;
}

class _ColumnCluster {
  const _ColumnCluster({
    required this.start,
    required this.end,
    required this.maxCount,
    required this.area,
  });

  final int start;
  final int end;
  final int maxCount;
  final int area;

  int get width => end - start;
}
