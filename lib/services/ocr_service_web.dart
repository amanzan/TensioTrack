import 'dart:async';
import 'dart:convert';
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:google_generative_ai/google_generative_ai.dart';
import 'package:http/http.dart' as http;

import 'ocr_service.dart';

/// Retorna la instancia de servicio OCR web (Gemini Vision).
OcrService getOcrService() => GeminiWebOcrService();

class GeminiWebOcrService implements OcrService {
  /// Clave API de Gemini (misma configuración que en móvil).
  static const _geminiApiKey = String.fromEnvironment('GEMINI_API_KEY');
  static const _groqApiKey = String.fromEnvironment('GROQ_API_KEY');
  static const _githubModelsToken = String.fromEnvironment(
    'GITHUB_MODELS_TOKEN',
  );
  static const _githubModelsModel = String.fromEnvironment(
    'GITHUB_MODELS_MODEL',
    defaultValue: 'openai/gpt-4o-mini',
  );
  static const _forceGithubOcr = bool.fromEnvironment('FORCE_GITHUB_OCR');
  static const _forceGroqOcr = bool.fromEnvironment('FORCE_GROQ_OCR');
  static const _timeout = Duration(seconds: 60);

  @override
  Future<OcrResult?> recognizePressure(
    String imagePath,
    Uint8List imageBytes,
  ) async {
    try {
      debugPrint(
        _forceGithubOcr
            ? 'TensioTrack Web: Modo cloud forzado a GitHub Models.'
            : _forceGroqOcr
            ? 'TensioTrack Web: Modo cloud forzado a Groq.'
            : 'TensioTrack Web: Modo cloud Gemini -> GitHub Models -> Gemini -> GitHub Models -> Groq -> Groq.',
      );
      return _recognizeWithCloudProvidersWithRetries(imageBytes);
    } catch (e) {
      debugPrint('TensioTrack Web OCR ERROR: $e');
      rethrow;
    }
  }

  Future<OcrResult?> _recognizeWithGemini(Uint8List imageBytes) async {
    if (_geminiApiKey.isEmpty) {
      debugPrint(
        'TensioTrack Web: API key de Gemini no configurada. '
        'Se probará GitHub Models si tiene token.',
      );
      return null;
    }

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

    debugPrint('TensioTrack Web Gemini: Enviando imagen...');
    final response = await model.generateContent([content]).timeout(_timeout);
    final text = response.text?.trim();

    debugPrint('TensioTrack Web Gemini respuesta: "$text"');

    if (text == null || text.isEmpty) return null;
    final values = _parseGeminiTextResponse(text);
    if (values == null) return null;

    debugPrint(
      'TensioTrack Web Gemini RESULTADO: SYS=${values.systolic}, DIA=${values.diastolic}',
    );

    return _buildCloudResult(
      imageBytes: imageBytes,
      systolic: values.systolic,
      diastolic: values.diastolic,
      confidence: (values.systolic > 0 && values.diastolic > 0) ? 0.95 : 0.6,
      engineName: 'Gemini Vision (Web)',
    );
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
          'TensioTrack Web ${provider.label}: intento $attempt/${providerPlan.length}...',
        );
        final result = switch (provider) {
          _CloudOcrProvider.gemini => await _recognizeWithGemini(imageBytes),
          _CloudOcrProvider.github => await _recognizeWithGithub(imageBytes),
          _CloudOcrProvider.groq => await _recognizeWithGroq(imageBytes),
        };
        if (result != null) return result;
      } catch (e) {
        lastError = e;
        debugPrint(
          'TensioTrack Web ${provider.label}: intento $attempt/${providerPlan.length} falló ($e).',
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
    if (_forceGithubOcr) {
      return const [_CloudOcrProvider.github, _CloudOcrProvider.github];
    }
    if (_forceGroqOcr) {
      return const [_CloudOcrProvider.groq, _CloudOcrProvider.groq];
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

  Future<OcrResult?> _recognizeWithGithub(Uint8List imageBytes) async {
    if (_githubModelsToken.isEmpty) {
      debugPrint(
        'TensioTrack Web: token de GitHub Models no configurado. '
        'Ejecuta con: flutter run --dart-define-from-file=.env.json',
      );
      return null;
    }

    debugPrint(
      'TensioTrack Web GitHub Models: Enviando imagen '
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
        'TensioTrack Web GitHub Models ERROR: ${response.statusCode} ${response.body}',
      );
      return null;
    }

    final responseJson = jsonDecode(response.body) as Map<String, dynamic>;
    final choices = responseJson['choices'] as List<dynamic>?;
    final message = choices?.isNotEmpty == true
        ? choices!.first['message'] as Map<String, dynamic>?
        : null;
    final content = message?['content'] as String?;

    debugPrint('TensioTrack Web GitHub Models respuesta: "$content"');

    if (content == null || content.trim().isEmpty) return null;
    final values = _parseCloudBpResponse(content);
    if (values == null) return null;

    debugPrint(
      'TensioTrack Web GitHub Models RESULTADO: SYS=${values.systolic}, DIA=${values.diastolic}',
    );

    return _buildCloudResult(
      imageBytes: imageBytes,
      systolic: values.systolic,
      diastolic: values.diastolic,
      confidence: 0.93,
      engineName: 'GitHub Models ($_githubModelsModel)',
    );
  }

  Future<OcrResult?> _recognizeWithGroq(Uint8List imageBytes) async {
    if (_groqApiKey.isEmpty) {
      debugPrint(
        'TensioTrack Web: API key de Groq no configurada. '
        'Ejecuta con: flutter run --dart-define-from-file=.env.json',
      );
      return null;
    }

    debugPrint(
      'TensioTrack Web Groq: Enviando imagen '
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
        'TensioTrack Web Groq ERROR: ${response.statusCode} ${response.body}',
      );
      return null;
    }

    final responseJson = jsonDecode(response.body) as Map<String, dynamic>;
    final choices = responseJson['choices'] as List<dynamic>?;
    final message = choices?.isNotEmpty == true
        ? choices!.first['message'] as Map<String, dynamic>?
        : null;
    final content = message?['content'] as String?;

    debugPrint('TensioTrack Web Groq respuesta: "$content"');

    if (content == null || content.trim().isEmpty) return null;
    final values = _parseCloudBpResponse(content);
    if (values == null) return null;

    debugPrint(
      'TensioTrack Web Groq RESULTADO: SYS=${values.systolic}, DIA=${values.diastolic}',
    );

    return _buildCloudResult(
      imageBytes: imageBytes,
      systolic: values.systolic,
      diastolic: values.diastolic,
      confidence: 0.92,
      engineName: 'Groq Llama Vision (Web)',
    );
  }

  _PressureValues? _parseGeminiTextResponse(String text) {
    int sys = 0, dia = 0;
    final parts = text.split(',');

    if (parts.length >= 2) {
      sys = int.tryParse(parts[0].trim()) ?? 0;
      dia = int.tryParse(parts[1].trim()) ?? 0;
    } else {
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
    return _PressureValues(systolic: sys, diastolic: dia);
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

class _PressureValues {
  const _PressureValues({required this.systolic, required this.diastolic});

  final int systolic;
  final int diastolic;
}

enum _CloudOcrProvider {
  gemini('Gemini'),
  github('GitHub Models'),
  groq('Groq');

  const _CloudOcrProvider(this.label);

  final String label;
}
