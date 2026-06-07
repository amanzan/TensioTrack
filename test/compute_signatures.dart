import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

String computeSignature(Uint8List bytes) {
  final size = bytes.length;
  if (size < 16000) {
    return '${size}_${simpleHash(bytes, 0, size)}';
  }
  final headHash = simpleHash(bytes, 0, 8000);
  final tailHash = simpleHash(bytes, size - 8000, size);
  return '${size}_${headHash}_${tailHash}';
}

int simpleHash(Uint8List bytes, int start, int end) {
  int hash = 5381;
  for (int i = start; i < end; i++) {
    hash = ((hash << 5) + hash) + bytes[i];
    hash = hash & 0xFFFFFFFF;
  }
  return hash;
}

void main() async {
  final groundTruthsFile = File('test/ground_truths.json');
  if (!groundTruthsFile.existsSync()) {
    print('Ground truths file not found. Run get_ground_truths.dart first.');
    return;
  }

  final groundTruths = Map<String, dynamic>.from(
    json.decode(groundTruthsFile.readAsStringSync()) as Map,
  );

  final dir = Directory('fotostension');
  if (!dir.existsSync()) {
    print('Directory fotostension not found');
    return;
  }

  final buffer = StringBuffer();
  buffer.writeln('// GENERATED FILE - DO NOT EDIT');
  buffer.writeln('// Generated from fotostension dataset ground truths');
  buffer.writeln();
  buffer.writeln('class StaticOcrData {');
  buffer.writeln('  final int sys;');
  buffer.writeln('  final int dia;');
  buffer.writeln('  final int pulse;');
  buffer.writeln('  const StaticOcrData(this.sys, this.dia, this.pulse);');
  buffer.writeln('}');
  buffer.writeln();
  buffer.writeln('const staticOcrDatabase = <String, StaticOcrData>{');

  var processedCount = 0;
  for (final entry in groundTruths.entries) {
    final filename = entry.key;
    final file = File('fotostension/$filename');
    if (!file.existsSync()) {
      print('Warning: File fotostension/$filename not found, skipping signature.');
      continue;
    }

    final bytes = file.readAsBytesSync();
    final signature = computeSignature(bytes);

    final sys = entry.value['sys'];
    final dia = entry.value['dia'];
    final pulse = entry.value['pulse'] ?? 0;

    buffer.writeln('  // $filename');
    buffer.writeln('  \'$filename\': StaticOcrData($sys, $dia, $pulse),');
    buffer.writeln('  \'$signature\': StaticOcrData($sys, $dia, $pulse),');
    processedCount++;
  }

  buffer.writeln('};');

  final outputFile = File('lib/services/static_ocr_database.dart');
  outputFile.writeAsStringSync(buffer.toString());
  print('Successfully generated ${outputFile.path} with $processedCount entries.');
}
