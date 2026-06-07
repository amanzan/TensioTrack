import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter_test/flutter_test.dart';
import 'package:tensiotrack/services/ocr_service.dart';
import 'package:tensiotrack/services/ocr_service_mobile.dart';

Future<Uint8List> resizeImageNative(Uint8List bytes, int maxDimension) async {
  final codec = await ui.instantiateImageCodec(bytes);
  final frame = await codec.getNextFrame();
  final src = frame.image;

  final double scale = maxDimension / math.max(src.width, src.height);
  if (scale >= 1.0) {
    return bytes;
  }

  final int destWidth = (src.width * scale).round();
  final int destHeight = (src.height * scale).round();

  final recorder = ui.PictureRecorder();
  final canvas = ui.Canvas(recorder);
  
  canvas.drawImageRect(
    src,
    ui.Rect.fromLTWH(0, 0, src.width.toDouble(), src.height.toDouble()),
    ui.Rect.fromLTWH(0, 0, destWidth.toDouble(), destHeight.toDouble()),
    ui.Paint()..filterQuality = ui.FilterQuality.high,
  );
  
  final picture = recorder.endRecording();
  final resized = await picture.toImage(destWidth, destHeight);
  
  final byteData = await resized.toByteData(format: ui.ImageByteFormat.png);
  return byteData!.buffer.asUint8List();
}

void main() {
  testWidgets('Analizar precision de todas las fotos offline', (tester) async {
    TestWidgetsFlutterBinding.ensureInitialized();

    // Envolver en runAsync para permitir operaciones asíncronas reales (I/O) en el entorno de test de Flutter
    await tester.runAsync(() async {
      GeminiOcrService.bypassStaticLookup = true;
      final dir = Directory('fotostension');
      if (!await dir.exists()) {
        fail('El directorio fotostension no existe.');
      }

      final files = dir.listSync()
          .whereType<File>()
          .where((file) {
            final p = file.path.toLowerCase();
            return p.endsWith('.png') || p.endsWith('.jpg') || p.endsWith('.jpeg');
          })
          .toList();

      final resultFile = File('test_results.txt');
      if (await resultFile.exists()) {
        await resultFile.delete();
      }
      final sink = resultFile.openWrite();

      sink.writeln('======================================================');
      sink.writeln('PROCESANDO ${files.length} IMÁGENES DE FOTOSTENSION (NATIVE RESIZED TO 400)...');
      sink.writeln('======================================================');
      print('Iniciando procesamiento nativo de ${files.length} imágenes (max 400px)...');

      final ocrService = OcrService();
      var successCount = 0;

      for (final file in files) {
        final name = file.path.split('/').last;
        var bytes = await file.readAsBytes();

        // Redimensionar usando motor nativo de Flutter
        try {
          bytes = await resizeImageNative(bytes, 400);
        } catch (e) {
          sink.writeln('FOTO: $name => ERROR REDIMENSIONANDO NATIVAMENTE: $e');
        }

        try {
          final result = await ocrService.recognizePressure(file.path, bytes, useAlternate: false);
          if (result != null) {
            final logMsg = 'FOTO: $name => SYS: ${result.systolic}, DIA: ${result.diastolic} (conf: ${result.confidence.toStringAsFixed(2)}, motor: ${result.engineName})';
            sink.writeln(logMsg);
            print(logMsg);
            successCount++;
          } else {
            final logMsg = 'FOTO: $name => FALLÓ (null)';
            sink.writeln(logMsg);
            print(logMsg);
          }
        } catch (e) {
          final logMsg = 'FOTO: $name => ERROR: $e';
          sink.writeln(logMsg);
          print(logMsg);
        }
        await sink.flush();
      }

      sink.writeln('======================================================');
      sink.writeln('PROCESO TERMINADO. EXITO: $successCount/${files.length}');
      sink.writeln('======================================================');
      await sink.close();
      print('Procesamiento completo. Resultados guardados en test_results.txt');
    });
  });
}
