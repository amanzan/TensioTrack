// ignore_for_file: avoid_print

import 'dart:convert';
import 'dart:io';

Future<void> main(List<String> args) async {
  final options = _Options.parse(args);
  final dir = Directory(options.dirPath);
  if (!dir.existsSync()) {
    stderr.writeln('Dataset no encontrado: ${options.dirPath}');
    exitCode = 1;
    return;
  }

  final files =
      dir
          .listSync()
          .whereType<File>()
          .where(
            (file) =>
                RegExp(
                  r'\.(jpe?g|png)$',
                  caseSensitive: false,
                ).hasMatch(file.path) &&
                _expectedReadingFromFile(file) != null,
          )
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));

  final server = await HttpServer.bind(options.host, options.port);
  print('SERVING dir=${dir.absolute.path}');
  print('LISTENING http://${options.host}:${options.port}/');
  print('IMAGES ${files.length}');

  await for (final request in server) {
    try {
      await _handleRequest(request, files);
    } catch (error, stackTrace) {
      stderr.writeln('ERROR $error\n$stackTrace');
      request.response
        ..statusCode = HttpStatus.internalServerError
        ..write('Internal error')
        ..close();
    }
  }
}

Future<void> _handleRequest(HttpRequest request, List<File> files) async {
  final path = request.uri.path;

  if (path == '/' || path == '/manifest.json') {
    final items = [
      for (final file in files)
        {
          'file': file.uri.pathSegments.last,
          'expectedSys': _expectedReadingFromFile(file)!.$1,
          'expectedDia': _expectedReadingFromFile(file)!.$2,
          'url': '/images/${Uri.encodeComponent(file.uri.pathSegments.last)}',
        },
    ];
    final body = jsonEncode({'total': items.length, 'images': items});
    request.response
      ..headers.contentType = ContentType.json
      ..write(body);
    await request.response.close();
    return;
  }

  if (path.startsWith('/images/')) {
    final name = Uri.decodeComponent(path.substring('/images/'.length));
    final file = files.cast<File?>().firstWhere(
      (item) => item!.uri.pathSegments.last == name,
      orElse: () => null,
    );
    if (file == null) {
      request.response.statusCode = HttpStatus.notFound;
      await request.response.close();
      return;
    }

    request.response.headers.contentType = _contentTypeFor(file.path);
    await request.response.addStream(file.openRead());
    await request.response.close();
    return;
  }

  request.response.statusCode = HttpStatus.notFound;
  await request.response.close();
}

ContentType _contentTypeFor(String path) {
  final lower = path.toLowerCase();
  if (lower.endsWith('.png')) return ContentType('image', 'png');
  return ContentType('image', 'jpeg');
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

  return null;
}

class _Options {
  const _Options({
    required this.dirPath,
    required this.host,
    required this.port,
  });

  final String dirPath;
  final String host;
  final int port;

  static _Options parse(List<String> args) {
    var dirPath = '../fotostension';
    var host = '127.0.0.1';
    var port = 8787;

    for (var index = 0; index < args.length; index++) {
      switch (args[index]) {
        case '--dir':
          dirPath = args[++index];
        case '--host':
          host = args[++index];
        case '--port':
          port = int.parse(args[++index]);
      }
    }

    return _Options(dirPath: dirPath, host: host, port: port);
  }
}
