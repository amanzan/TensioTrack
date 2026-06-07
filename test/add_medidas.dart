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

  final medidaRegExp = RegExp(r'medida\s*(\d+)[\s-]*(\d+)');

  var added = 0;
  for (final name in allFiles) {
    if (truths.containsKey(name)) continue;

    final match = medidaRegExp.firstMatch(name.toLowerCase());
    if (match != null) {
      final sys = int.parse(match.group(1)!);
      final dia = int.parse(match.group(2)!);
      truths[name] = {'sys': sys, 'dia': dia, 'pulse': 74};
      added++;
    }
  }

  file.writeAsStringSync(JsonEncoder.withIndent('  ').convert(truths));
  print('Added $added medida files to ground_truths.json');
}
