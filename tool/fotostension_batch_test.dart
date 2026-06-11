// ignore_for_file: avoid_print

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tensiotrack/services/blood_pressure_digit_reader.dart';

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

    final reader = BloodPressureDigitReader(modelAssetPath: modelAsset);
    var ok = 0;
    var failed = 0;

    print('DIR=$dirPath');
    print('MODEL=$modelAsset');
    print('file,expected_sys,expected_dia,detected_sys,detected_dia,status');

    for (final file in files) {
      final expected = _expectedReadingFromFile(file);
      if (expected == null) {
        print('${file.path},,,,,BAD_NAME');
        failed++;
        continue;
      }

      final (expectedSys, expectedDia) = expected;
      final result = await reader.readFromImagePath(file.path);
      final detectedSys = result?.systolic;
      final detectedDia = result?.diastolic;
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
        '${detectedSys ?? ''},${detectedDia ?? ''},$status',
      );
    }

    await reader.dispose();
    print('SUMMARY,total=${files.length},ok=$ok,failed=$failed');
  }, timeout: const Timeout(Duration(minutes: 5)));
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
