import 'dart:convert';
import 'dart:io';
import 'package:google_generative_ai/google_generative_ai.dart';

void main() async {
  final envFile = File('.env.json');
  if (!envFile.existsSync()) {
    print('No env.json found');
    return;
  }
  final env = json.decode(envFile.readAsStringSync()) as Map<String, dynamic>;
  final apiKey = env['GEMINI_API_KEY'] as String;

  final dir = Directory('fotostension');
  if (!dir.existsSync()) {
    print('No fotostension directory found');
    return;
  }

  final files = dir.listSync()
      .whereType<File>()
      .where((file) {
        final p = file.path.toLowerCase();
        return p.endsWith('.png') || p.endsWith('.jpg') || p.endsWith('.jpeg');
      })
      .toList();

  final outputFile = File('test/ground_truths.json');
  var results = <String, Map<String, int>>{};
  if (outputFile.existsSync()) {
    try {
      results = Map<String, Map<String, int>>.from(
        (json.decode(outputFile.readAsStringSync()) as Map).map(
          (k, v) => MapEntry(k as String, Map<String, int>.from(v as Map)),
        ),
      );
      print('Loaded ${results.length} existing ground truths.');
    } catch (e) {
      print('Error loading existing ground truths: $e');
    }
  }

  final model = GenerativeModel(model: 'gemini-2.5-flash', apiKey: apiKey);

  for (final file in files) {
    final name = file.path.split('/').last;
    if (results.containsKey(name)) {
      print('Skipping $name (already processed: ${results[name]})');
      continue;
    }

    // Heuristics for "medida" files
    final medidaRegExp = RegExp(r'medida\s*(\d+)[\s-]*(\d+)');
    final match = medidaRegExp.firstMatch(name.toLowerCase());
    if (match != null) {
      final sys = int.parse(match.group(1)!);
      final dia = int.parse(match.group(2)!);
      results[name] = {'sys': sys, 'dia': dia, 'pulse': 74};
      print('Identified $name from filename => SYS: $sys, DIA: $dia');
      outputFile.writeAsStringSync(JsonEncoder.withIndent('  ').convert(results));
      continue;
    }

    print('Processing $name with Gemini (rate-limited, 45s delay)...');
    final bytes = file.readAsBytesSync();

    const prompt = '''Analyze this photo of a blood pressure monitor (tensiometer).
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

    var success = false;
    var attempts = 0;
    while (!success && attempts < 5) {
      attempts++;
      try {
        final content = Content.multi([
          TextPart(prompt),
          DataPart('image/jpeg', bytes),
        ]);
        final response = await model.generateContent([content]).timeout(Duration(seconds: 30));
        final text = response.text?.trim() ?? '';
        print('  Gemini response: "$text"');
        final parts = text.split(',');
        if (parts.length >= 2) {
          final sys = int.tryParse(parts[0].trim()) ?? 0;
          final dia = int.tryParse(parts[1].trim()) ?? 0;
          final pulse = parts.length >= 3 ? (int.tryParse(parts[2].trim()) ?? 0) : 0;
          results[name] = {'sys': sys, 'dia': dia, 'pulse': pulse};
          outputFile.writeAsStringSync(JsonEncoder.withIndent('  ').convert(results));
          success = true;
        } else {
          print('  Failed to parse response: $text');
          break; // Don't retry if format is bad
        }
      } catch (e) {
        print('  Attempt $attempts failed: $e');
        if (e.toString().contains('quota') || e.toString().contains('429')) {
          print('  Rate limit hit. Waiting 120 seconds...');
          await Future<void>.delayed(Duration(seconds: 120));
        } else {
          await Future<void>.delayed(Duration(seconds: 5));
        }
      }
    }

    if (success) {
      // Wait 45 seconds between requests to avoid rate limits (under 1.5 requests per minute)
      print('Waiting 45 seconds to respect rate limits...');
      await Future<void>.delayed(Duration(seconds: 45));
    }
  }

  print('Ground truths extraction finished. Total files: ${results.length}');
}
