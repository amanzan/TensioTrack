// ignore_for_file: avoid_print

import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_generative_ai/google_generative_ai.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const _RunnerApp());
}

class _RunnerApp extends StatefulWidget {
  const _RunnerApp();

  @override
  State<_RunnerApp> createState() => _RunnerAppState();
}

class _RunnerAppState extends State<_RunnerApp> {
  static const serverUrl = String.fromEnvironment(
    'IMAGE_SERVER',
    defaultValue: 'http://127.0.0.1:8787',
  );
  static const engineName = String.fromEnvironment(
    'ENGINE',
    defaultValue: 'groq',
  );
  static const maxImages = int.fromEnvironment('MAX_IMAGES', defaultValue: 3);
  static const onlyFile = String.fromEnvironment('ONLY_FILE');
  static const delayMs = int.fromEnvironment('DELAY_MS', defaultValue: 5000);
  static const maxAttempts = int.fromEnvironment(
    'MAX_ATTEMPTS',
    defaultValue: 1,
  );
  static const initialBackoffMs = int.fromEnvironment(
    'INITIAL_BACKOFF_MS',
    defaultValue: 15000,
  );
  static const maxBackoffMs = int.fromEnvironment(
    'MAX_BACKOFF_MS',
    defaultValue: 120000,
  );
  static const stopOnRateLimit = bool.fromEnvironment('STOP_ON_RATE_LIMIT');
  static const maxDim = int.fromEnvironment('MAX_DIM', defaultValue: 1600);
  static const jpegQuality = int.fromEnvironment(
    'JPEG_QUALITY',
    defaultValue: 85,
  );

  final _lines = <String>['Preparando prueba cloud en dispositivo...'];

  @override
  void initState() {
    super.initState();
    Future<void>.microtask(_run);
  }

  Future<void> _run() async {
    final engine = _cloudEngineFromName(engineName);
    var ok = 0;
    var failed = 0;
    var errors = 0;
    var processed = 0;
    final durationsMs = <int>[];

    void log(String line) {
      print(line);
      if (!mounted) return;
      setState(() => _lines.add(line));
    }

    try {
      log('DEVICE_CLOUD_RUNNER_START');
      log('IMAGE_SERVER=$serverUrl');
      log('ENGINE=${engine.name}');
      if (engine == _CloudEngine.github) {
        log('GITHUB_MODELS_MODEL=$_githubModelsModel');
      }
      log('MAX_IMAGES=$maxImages');
      if (onlyFile.isNotEmpty) log('ONLY_FILE=$onlyFile');
      log('DELAY_MS=$delayMs');
      log('MAX_ATTEMPTS=$maxAttempts');
      log('INITIAL_BACKOFF_MS=$initialBackoffMs');
      log('MAX_BACKOFF_MS=$maxBackoffMs');
      log('STOP_ON_RATE_LIMIT=$stopOnRateLimit');
      log('MAX_DIM=$maxDim');
      log('JPEG_QUALITY=$jpegQuality');
      log(
        'file,expected_sys,expected_dia,detected_sys,detected_dia,engine,status,duration_ms,detail',
      );

      final baseUri = Uri.parse(serverUrl);
      final manifestUri = baseUri.resolve('/manifest.json');
      final manifestResponse = await http
          .get(manifestUri)
          .timeout(const Duration(seconds: 20));
      if (manifestResponse.statusCode != 200) {
        throw StateError(
          'Manifest HTTP ${manifestResponse.statusCode}: ${manifestResponse.body}',
        );
      }
      final manifest =
          jsonDecode(manifestResponse.body) as Map<String, dynamic>;
      final images = (manifest['images'] as List<dynamic>)
          .cast<Map<String, dynamic>>();
      final filtered = onlyFile.isEmpty
          ? images
          : images
                .where((item) => item['file'] == onlyFile)
                .toList(growable: false);
      if (onlyFile.isNotEmpty && filtered.isEmpty) {
        throw StateError('No existe la imagen solicitada: $onlyFile');
      }
      final selected = maxImages <= 0
          ? filtered
          : filtered.take(maxImages).toList();

      for (var index = 0; index < selected.length; index++) {
        var stopRun = false;
        final item = selected[index];
        final file = item['file'] as String;
        final expectedSys = item['expectedSys'] as int;
        final expectedDia = item['expectedDia'] as int;
        final imageUri = baseUri.resolve(item['url'] as String);

        int? detectedSys;
        int? detectedDia;
        int? durationMs;
        var status = 'ERROR';
        var detail = '';

        try {
          final imageResponse = await http
              .get(imageUri)
              .timeout(const Duration(seconds: 30));
          if (imageResponse.statusCode != 200) {
            throw StateError('Image HTTP ${imageResponse.statusCode}');
          }
          final imageBytes = _prepareCloudBytes(
            imageResponse.bodyBytes,
            maxDim: maxDim,
            jpegQuality: jpegQuality,
          );
          final timedResult = await _recognizeTimedWithBackoff(
            engine,
            imageBytes,
            maxAttempts: maxAttempts,
            initialBackoffMs: initialBackoffMs,
            maxBackoffMs: maxBackoffMs,
            log: log,
          );
          final values = timedResult.values;
          durationMs = timedResult.durationMs;
          durationsMs.add(durationMs);
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
        } catch (error) {
          errors++;
          status = 'ERROR';
          detail = '${error.runtimeType}: $error';
          stopRun = stopOnRateLimit && _isRateLimitError(error);
        }
        processed++;

        log(
          '$file,$expectedSys,$expectedDia,${detectedSys ?? ''},${detectedDia ?? ''},'
          '${engine.name},$status,${durationMs ?? ''},"${_csvEscape(detail)}"',
        );

        if (stopRun) {
          log('STOPPED_ON_RATE_LIMIT file=$file');
          break;
        }

        if (delayMs > 0 && index < selected.length - 1) {
          await Future<void>.delayed(Duration(milliseconds: delayMs));
        }
      }

      log('SUMMARY,total=$processed,ok=$ok,failed=$failed,errors=$errors');
      log(_timingSummary(durationsMs));
      log('DEVICE_CLOUD_RUNNER_DONE');
    } catch (error, stackTrace) {
      log('RUNNER_FATAL ${error.runtimeType}: $error');
      log(stackTrace.toString());
    } finally {
      await Future<void>.delayed(const Duration(seconds: 2));
      await SystemNavigator.pop();
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(title: const Text('TensioTrack Cloud Runner')),
        body: ListView.builder(
          padding: const EdgeInsets.all(12),
          itemCount: _lines.length,
          itemBuilder: (context, index) => Text(
            _lines[index],
            style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
          ),
        ),
      ),
    );
  }
}

enum _CloudEngine { gemini, github, groq }

class _PressureValues {
  const _PressureValues({required this.systolic, required this.diastolic});

  final int systolic;
  final int diastolic;
}

class _TimedRecognition {
  const _TimedRecognition({required this.values, required this.durationMs});

  final _PressureValues? values;
  final int durationMs;
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

Future<_TimedRecognition> _recognizeTimedWithBackoff(
  _CloudEngine engine,
  Uint8List imageBytes, {
  required int maxAttempts,
  required int initialBackoffMs,
  required int maxBackoffMs,
  required void Function(String line) log,
}) async {
  Object? lastError;
  final attempts = maxAttempts < 1 ? 1 : maxAttempts;
  var nextBackoffMs = initialBackoffMs < 0 ? 0 : initialBackoffMs;
  final cappedMaxBackoffMs = maxBackoffMs < 0 ? 0 : maxBackoffMs;

  for (var attempt = 1; attempt <= attempts; attempt++) {
    final stopwatch = Stopwatch()..start();
    try {
      final values = await _recognizeDirect(engine, imageBytes);
      stopwatch.stop();
      return _TimedRecognition(
        values: values,
        durationMs: stopwatch.elapsedMilliseconds,
      );
    } catch (error) {
      stopwatch.stop();
      lastError = error;
      final canRetry = attempt < attempts && _isRateLimitError(error);
      if (!canRetry) rethrow;

      final retryMs = _retryDelayFromError(error) ?? nextBackoffMs;
      log(
        'BACKOFF,engine=${engine.name},attempt=$attempt,error=${_csvEscape(error.toString())},wait_ms=$retryMs',
      );
      if (retryMs > 0) {
        await Future<void>.delayed(Duration(milliseconds: retryMs));
      }
      nextBackoffMs = (nextBackoffMs * 2).clamp(0, cappedMaxBackoffMs);
    }
  }

  throw lastError ?? StateError('Reconocimiento no ejecutado');
}

Future<_PressureValues?> _recognizeWithGemini(Uint8List imageBytes) async {
  if (_geminiApiKey.isEmpty) throw StateError('GEMINI_API_KEY no configurada');

  final model = GenerativeModel(
    model: 'gemini-2.5-flash-lite',
    apiKey: _geminiApiKey,
  );
  const prompt =
      'Analyze this photo of a blood pressure monitor. Return ONLY three integers '
      'separated by commas in this exact format: systolic,diastolic,pulse. '
      'Use 0 for unclear values. No explanation.';

  final response = await model
      .generateContent([
        Content.multi([TextPart(prompt), DataPart('image/jpeg', imageBytes)]),
      ])
      .timeout(_timeout);

  final text = response.text?.trim();
  if (text == null || text.isEmpty) return null;
  return _parseGeminiTextResponse(text);
}

Future<_PressureValues?> _recognizeWithGithub(Uint8List imageBytes) async {
  if (_githubModelsToken.isEmpty) {
    throw StateError('GITHUB_MODELS_TOKEN no configurado');
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
                    'url': 'data:image/jpeg;base64,${base64Encode(imageBytes)}',
                  },
                },
                {
                  'type': 'text',
                  'text':
                      'This is a photo of a blood pressure monitor. Return ONLY JSON '
                      'like {"systolic": 120, "diastolic": 80, "pulse": 65}.',
                },
              ],
            },
          ],
        }),
      )
      .timeout(_timeout);

  if (response.statusCode != 200) {
    throw StateError(
      'GitHub HTTP ${response.statusCode}: ${_short(response.body)}',
    );
  }

  final decoded = jsonDecode(response.body) as Map<String, dynamic>;
  final choices = decoded['choices'] as List<dynamic>?;
  final message = choices?.isNotEmpty == true
      ? choices!.first['message'] as Map<String, dynamic>?
      : null;
  final content = message?['content'] as String?;
  if (content == null || content.trim().isEmpty) return null;
  return _parseCloudJsonResponse(content);
}

Future<_PressureValues?> _recognizeWithGroq(Uint8List imageBytes) async {
  if (_groqApiKey.isEmpty) throw StateError('GROQ_API_KEY no configurada');

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
                    'url': 'data:image/jpeg;base64,${base64Encode(imageBytes)}',
                  },
                },
                {
                  'type': 'text',
                  'text':
                      'This is a photo of a blood pressure monitor. Return ONLY a JSON '
                      'object like {"systolic": 120, "diastolic": 80}. No markdown.',
                },
              ],
            },
          ],
        }),
      )
      .timeout(_timeout);

  if (response.statusCode != 200) {
    throw StateError(
      'Groq HTTP ${response.statusCode}: ${_short(response.body)}',
    );
  }

  final decoded = jsonDecode(response.body) as Map<String, dynamic>;
  final choices = decoded['choices'] as List<dynamic>?;
  final message = choices?.isNotEmpty == true
      ? choices!.first['message'] as Map<String, dynamic>?
      : null;
  final content = message?['content'] as String?;
  if (content == null || content.trim().isEmpty) return null;
  return _parseCloudJsonResponse(content);
}

_PressureValues? _parseGeminiTextResponse(String text) {
  final numbers = RegExp(r'\d+')
      .allMatches(text)
      .map((match) => int.parse(match.group(0)!))
      .where((number) => number >= 30 && number <= 260)
      .toList();
  if (numbers.length < 2) return null;
  var sys = numbers[0];
  var dia = numbers[1];
  if (sys < dia) {
    final temp = sys;
    sys = dia;
    dia = temp;
  }
  return _PressureValues(systolic: sys, diastolic: dia);
}

_PressureValues? _parseCloudJsonResponse(String content) {
  try {
    final clean = content.replaceAll(RegExp(r'```json|```'), '').trim();
    final decoded = jsonDecode(clean);
    if (decoded is! Map<String, dynamic>) return null;
    var sys = (decoded['systolic'] as num?)?.toInt();
    var dia = (decoded['diastolic'] as num?)?.toInt();
    if (sys == null || dia == null) return null;
    if (sys < 40 || sys > 260 || dia < 40 || dia > 260) return null;
    if (sys < dia) {
      final temp = sys;
      sys = dia;
      dia = temp;
    }
    return _PressureValues(systolic: sys, diastolic: dia);
  } catch (_) {
    return null;
  }
}

Uint8List _prepareCloudBytes(
  Uint8List originalBytes, {
  required int maxDim,
  required int jpegQuality,
}) {
  final decoded = img.decodeImage(originalBytes);
  if (decoded == null || maxDim <= 0) return originalBytes;
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

String _short(String value) {
  final normalized = value.replaceAll(RegExp(r'\s+'), ' ').trim();
  if (normalized.length <= 180) return normalized;
  return '${normalized.substring(0, 180)}...';
}

String _timingSummary(List<int> durationsMs) {
  if (durationsMs.isEmpty) {
    return 'TIMING,count=0,total_ms=0,mean_ms=0.00,min_ms=0,max_ms=0';
  }

  final total = durationsMs.reduce((a, b) => a + b);
  final min = durationsMs.reduce((a, b) => a < b ? a : b);
  final max = durationsMs.reduce((a, b) => a > b ? a : b);
  final mean = total / durationsMs.length;

  return 'TIMING,count=${durationsMs.length},total_ms=$total,'
      'mean_ms=${mean.toStringAsFixed(2)},min_ms=$min,max_ms=$max';
}

bool _isRateLimitError(Object error) {
  final text = error.toString().toLowerCase();
  return text.contains('429') ||
      text.contains('quota') ||
      text.contains('rate limit') ||
      text.contains('rate-limit') ||
      text.contains('too many requests') ||
      text.contains('retry in');
}

int? _retryDelayFromError(Object error) {
  final text = error.toString();
  final match = RegExp(
    r'retry in ([0-9]+(?:\.[0-9]+)?)s',
    caseSensitive: false,
  ).firstMatch(text);
  if (match == null) return null;
  final seconds = double.tryParse(match.group(1)!);
  if (seconds == null) return null;
  return (seconds * 1000).ceil() + 1000;
}

String _csvEscape(String value) => value.replaceAll('"', '""');
