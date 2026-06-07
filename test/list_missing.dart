import 'dart:convert';
import 'dart:io';

void main() {
  final file = File('test/ground_truths.json');
  final truths = file.existsSync()
      ? Map<String, dynamic>.from(json.decode(file.readAsStringSync()) as Map)
      : <String, dynamic>{};

  final dir = Directory('fotostension');
  final allFiles = dir.listSync()
      .whereType<File>()
      .map((f) => f.path.split('/').last)
      .where((name) => name.endsWith('.png') || name.endsWith('.jpg') || name.endsWith('.jpeg'))
      .toList();

  final missing = allFiles.where((name) => !truths.containsKey(name)).toList();
  print('Total missing: ${missing.length}');
  print('Missing files:');
  for (final m in missing) {
    print('  - $m');
  }
}
