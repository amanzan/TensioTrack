// ignore_for_file: avoid_print

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tensiotrack/services/blood_pressure_digit_reader.dart';
import 'package:tensiotrack/services/ocr_service.dart';
import 'package:tensiotrack/services/ocr_service_mobile.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('reporte batch fotostension', () async {
    const fileFilter = String.fromEnvironment('FILE');
    const dirPath = String.fromEnvironment(
      'DIR',
      defaultValue: '../fotostension',
    );
    const modelAsset = String.fromEnvironment(
      'MODEL',
      defaultValue: 'best_float32.tflite',
    );
    const engineName = String.fromEnvironment('ENGINE', defaultValue: 'yolo');
    const recursive = bool.fromEnvironment('RECURSIVE');
    const verbose = bool.fromEnvironment('VERBOSE');

    if (!verbose) {
      debugPrint = (String? message, {int? wrapWidth}) {};
    }

    final dir = Directory(dirPath);
    final files =
        dir.existsSync()
              ? (dir
                    .listSync(recursive: recursive)
                    .whereType<File>()
                    .where(
                      (file) => RegExp(
                        r'\.(jpe?g|png)$',
                        caseSensitive: false,
                      ).hasMatch(file.path),
                    )
                    .where(
                      (file) =>
                          fileFilter.isEmpty || file.path.contains(fileFilter),
                    )
                    .toList())
              : <File>[]
          ..sort((a, b) => a.path.compareTo(b.path));

    final engine = _batchEngineFromName(engineName);
    final reader = engine == _BatchOcrEngine.yolo
        ? BloodPressureDigitReader(modelAssetPath: modelAsset)
        : null;
    OcrConfig.engine = switch (engine) {
      _BatchOcrEngine.yolo => OcrEngine.yolo,
      _BatchOcrEngine.hybrid => OcrEngine.hybrid,
    };
    final ocrService = engine == _BatchOcrEngine.yolo
        ? null
        : GeminiOcrService();
    var ok = 0;
    var failed = 0;
    final durationsMs = <int>[];

    print('DIR=$dirPath');
    print('ENGINE=${engine.name}');
    print('MODEL=$modelAsset');
    print(
      'file,expected_sys,expected_dia,detected_sys,detected_dia,status,duration_ms',
    );

    for (final file in files) {
      final expected = _expectedReadingFromFile(file);
      if (expected == null) {
        print('${file.path},,,,,BAD_NAME');
        failed++;
        continue;
      }

      final (expectedSys, expectedDia) = expected;
      final imageBytes = await file.readAsBytes();
      int? detectedSys;
      int? detectedDia;
      final stopwatch = Stopwatch()..start();
      switch (engine) {
        case _BatchOcrEngine.yolo:
          final result = await reader!.readFromImageBytes(imageBytes);
          detectedSys = result?.systolic;
          detectedDia = result?.diastolic;
        case _BatchOcrEngine.hybrid:
          final result = await ocrService!.recognizePressure(
            file.path,
            imageBytes,
          );
          detectedSys = result?.systolic;
          detectedDia = result?.diastolic;
      }
      stopwatch.stop();
      final durationMs = stopwatch.elapsedMilliseconds;
      durationsMs.add(durationMs);

      final status = detectedSys == expectedSys && detectedDia == expectedDia
          ? 'OK'
          : 'FAIL';

      if (status == 'OK') {
        ok++;
      } else {
        failed++;
      }

      print(
        '${file.path},$expectedSys,$expectedDia,'
        '${detectedSys ?? ''},${detectedDia ?? ''},$status,$durationMs',
      );
    }

    await reader?.dispose();
    print('SUMMARY,total=${files.length},ok=$ok,failed=$failed');
    print(_timingSummary(durationsMs));
  }, timeout: const Timeout(Duration(minutes: 5)));
}

enum _BatchOcrEngine { yolo, hybrid }

_BatchOcrEngine _batchEngineFromName(String name) {
  return switch (name.toLowerCase().trim()) {
    'hybrid' || 'hibrido' => _BatchOcrEngine.hybrid,
    _ => _BatchOcrEngine.yolo,
  };
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
