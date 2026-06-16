// ignore_for_file: avoid_print

import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_generative_ai/google_generative_ai.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test(
    'reporte batch cloud OCR directo',
    () async {
      const dirPath = String.fromEnvironment(
        'DIR',
        defaultValue: '../fotostension',
      );
      const engineName = String.fromEnvironment(
        'ENGINE',
        defaultValue: 'gemini',
      );
      const maxImages = int.fromEnvironment('MAX_IMAGES', defaultValue: 5);
      const delayMs = int.fromEnvironment('DELAY_MS', defaultValue: 2500);
      const maxDim = int.fromEnvironment('MAX_DIM', defaultValue: 1600);
      const jpegQuality = int.fromEnvironment('JPEG_QUALITY', defaultValue: 85);
      const fileFilter = String.fromEnvironment('FILE');

      final engine = _cloudEngineFromName(engineName);
      final dir = Directory(dirPath);
      final files =
          dir.existsSync()
                ? (dir
                      .listSync()
                      .whereType<File>()
                      .where(
                        (file) => RegExp(
                          r'\.(jpe?g|png)$',
                          caseSensitive: false,
                        ).hasMatch(file.path),
                      )
                      .where(
                        (file) =>
                            fileFilter.isEmpty ||
                            file.path.contains(fileFilter),
                      )
                      .toList())
                : <File>[]
            ..sort((a, b) => a.path.compareTo(b.path));

      final selectedFiles = maxImages <= 0
          ? files
          : files.take(maxImages).toList();

      var ok = 0;
      var failed = 0;
      var errors = 0;

      print('DIR=$dirPath');
      print('ENGINE=${engine.name}');
      print('MAX_IMAGES=$maxImages');
      print('DELAY_MS=$delayMs');
      print('MAX_DIM=$maxDim');
      print('JPEG_QUALITY=$jpegQuality');
      print(
        'file,expected_sys,expected_dia,detected_sys,detected_dia,'
        'engine,status,detail',
      );

      for (var index = 0; index < selectedFiles.length; index++) {
        final file = selectedFiles[index];
        final expected = _expectedReadingFromFile(file);
        if (expected == null) {
          print('${file.path},,,,,${engine.name},BAD_NAME,');
          errors++;
          continue;
        }

        final (expectedSys, expectedDia) = expected;
        final imageBytes = _prepareCloudBytes(
          await file.readAsBytes(),
          maxDim: maxDim,
          jpegQuality: jpegQuality,
        );

        int? detectedSys;
        int? detectedDia;
        var status = 'ERROR';
        var detail = '';

        try {
          final values = await _recognizeDirect(engine, imageBytes);
          detectedSys = values?.systolic;
          detectedDia = values?.diastolic;

          if (values == null) {
            status = 'FAIL';
            failed++;
            detail = 'Sin lectura parseable';
          } else if (detectedSys == expectedSys && detectedDia == expectedDia) {
            status = 'OK';
            ok++;
            detail = 'Lectura exacta';
          } else {
            status = 'FAIL';
            failed++;
            detail = 'Lectura distinta';
          }
        } on _CloudOcrException catch (error) {
          errors++;
          status = 'ERROR';
          detail = error.message;
        } catch (error) {
          errors++;
          status = 'ERROR';
          detail = '${error.runtimeType}: $error';
        }

        print(
          '${file.path},$expectedSys,$expectedDia,'
          '${detectedSys ?? ''},${detectedDia ?? ''},'
          '${engine.name},$status,"${_csvEscape(detail)}"',
        );

        if (delayMs > 0 && index < selectedFiles.length - 1) {
          await Future<void>.delayed(Duration(milliseconds: delayMs));
        }
      }

      print(
        'SUMMARY,total=${selectedFiles.length},ok=$ok,failed=$failed,'
        'errors=$errors',
      );
    },
    timeout: const Timeout(Duration(minutes: 15)),
  );
}

enum _CloudEngine { gemini, github, groq }

class _PressureValues {
  const _PressureValues({required this.systolic, required this.diastolic});

  final int systolic;
  final int diastolic;
}

class _CloudOcrException implements Exception {
  const _CloudOcrException(this.message);

  final String message;
}

const _geminiApiKey = String.fromEnvironment('GEMINI_API_KEY');
const _groqApiKey = String.fromEnvironment('GROQ_API_KEY');
const _githubModelsToken = String.fromEnvironment('GITHUB_MODELS_TOKEN');
const _githubModelsModel = String.fromEnvironment(
  'GITHUB_MODELS_MODEL',
  defaultValue: 'openai/gpt-4o-mini',
);
const _timeout = Duration(seconds: 60);

Future<_PressureValues?> _recognizeDirect(
  _CloudEngine engine,
  Uint8List imageBytes,
) {
  return switch (engine) {
    _CloudEngine.gemini => _recognizeWithGemini(imageBytes),
    _CloudEngine.github => _recognizeWithGithub(imageBytes),
    _CloudEngine.groq => _recognizeWithGroq(imageBytes),
  };
}

Future<_PressureValues?> _recognizeWithGemini(Uint8List imageBytes) async {
  if (_geminiApiKey.isEmpty) {
    throw const _CloudOcrException('GEMINI_API_KEY no configurada');
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

  final response = await model
      .generateContent([
        Content.multi([TextPart(prompt), DataPart('image/jpeg', imageBytes)]),
      ])
      .timeout(_timeout);

  final text = response.text?.trim();
  if (text == null || text.isEmpty) {
    throw const _CloudOcrException('Gemini devolvió respuesta vacía');
  }

  return _parseGeminiTextResponse(text);
}

Future<_PressureValues?> _recognizeWithGithub(Uint8List imageBytes) async {
  if (_githubModelsToken.isEmpty) {
    throw const _CloudOcrException('GITHUB_MODELS_TOKEN no configurado');
  }

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
    throw _CloudOcrException(
      'GitHub Models HTTP ${response.statusCode}: ${_short(response.body)}',
    );
  }

  final responseJson = jsonDecode(response.body) as Map<String, dynamic>;
  final choices = responseJson['choices'] as List<dynamic>?;
  final message = choices?.isNotEmpty == true
      ? choices!.first['message'] as Map<String, dynamic>?
      : null;
  final content = message?['content'] as String?;
  if (content == null || content.trim().isEmpty) return null;
  return _parseCloudJsonResponse(content);
}

Future<_PressureValues?> _recognizeWithGroq(Uint8List imageBytes) async {
  if (_groqApiKey.isEmpty) {
    throw const _CloudOcrException('GROQ_API_KEY no configurada');
  }

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
    throw _CloudOcrException(
      'Groq HTTP ${response.statusCode}: ${_short(response.body)}',
    );
  }

  final responseJson = jsonDecode(response.body) as Map<String, dynamic>;
  final choices = responseJson['choices'] as List<dynamic>?;
  final message = choices?.isNotEmpty == true
      ? choices!.first['message'] as Map<String, dynamic>?
      : null;
  final content = message?['content'] as String?;
  if (content == null || content.trim().isEmpty) return null;
  return _parseCloudJsonResponse(content);
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
        .map((match) => int.parse(match.group(0)!))
        .where((number) => number >= 30 && number <= 250)
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

_PressureValues? _parseCloudJsonResponse(String content) {
  try {
    final clean = content.replaceAll(RegExp(r'```json|```'), '').trim();
    final decoded = jsonDecode(clean);
    if (decoded is! Map<String, dynamic>) return null;

    var systolic = (decoded['systolic'] as num?)?.toInt();
    var diastolic = (decoded['diastolic'] as num?)?.toInt();
    if (systolic == null || diastolic == null) return null;
    if (systolic < 40 || systolic > 260) return null;
    if (diastolic < 40 || diastolic > 260) return null;

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

Uint8List _prepareCloudBytes(
  Uint8List originalBytes, {
  required int maxDim,
  required int jpegQuality,
}) {
  if (maxDim <= 0) return originalBytes;

  final decoded = img.decodeImage(originalBytes);
  if (decoded == null) return originalBytes;

  final oriented = img.bakeOrientation(decoded);
  final longestSide = oriented.width > oriented.height
      ? oriented.width
      : oriented.height;
  final resized = longestSide > maxDim
      ? img.copyResize(
          oriented,
          width: oriented.width >= oriented.height ? maxDim : null,
          height: oriented.height > oriented.width ? maxDim : null,
          interpolation: img.Interpolation.linear,
        )
      : oriented;

  return Uint8List.fromList(
    img.encodeJpg(resized, quality: jpegQuality.clamp(1, 100)),
  );
}

_CloudEngine _cloudEngineFromName(String name) {
  return switch (name.toLowerCase().trim()) {
    'github' || 'github_models' || 'github-models' => _CloudEngine.github,
    'groq' => _CloudEngine.groq,
    _ => _CloudEngine.gemini,
  };
}

(int sys, int dia)? _expectedReadingFromFile(File file) {
  final name = file.uri.pathSegments.last;

  final originalMatch = RegExp(
    r'^\d+_([0-9]+)_([0-9]+)\.(jpe?g|png)$',
    caseSensitive: false,
  ).firstMatch(name);
  if (originalMatch != null) {
    return (
      int.parse(originalMatch.group(1)!),
      int.parse(originalMatch.group(2)!),
    );
  }

  final syntheticMatch = RegExp(
    r'^synth_([0-9]+)_([0-9]+)\.(jpe?g|png)$',
    caseSensitive: false,
  ).firstMatch(name);
  if (syntheticMatch != null) {
    return (
      int.parse(syntheticMatch.group(1)!),
      int.parse(syntheticMatch.group(2)!),
    );
  }

  return null;
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

String _short(String value) {
  final normalized = value.replaceAll(RegExp(r'\s+'), ' ').trim();
  if (normalized.length <= 220) return normalized;
  return '${normalized.substring(0, 220)}...';
}

String _csvEscape(String value) => value.replaceAll('"', '""');
