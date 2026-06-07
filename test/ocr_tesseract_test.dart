import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

// Helper to resize image native in Flutter (unused but kept for API)
Future<Uint8List> _resizeImageNative(Uint8List bytes, int maxDimension) async {
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

Future<ui.Image> _cropImage(ui.Image src, ui.Rect cropRect) async {
  final recorder = ui.PictureRecorder();
  final canvas = ui.Canvas(recorder);
  final srcRect = cropRect;

  final double scale = 2.0;
  final destRect = ui.Rect.fromLTWH(
    0,
    0,
    cropRect.width * scale,
    cropRect.height * scale,
  );

  canvas.drawImageRect(
    src,
    srcRect,
    destRect,
    ui.Paint()..filterQuality = ui.FilterQuality.high,
  );
  final picture = recorder.endRecording();
  return await picture.toImage(
    (cropRect.width * scale).toInt(),
    (cropRect.height * scale).toInt(),
  );
}

Future<String> _saveImageToTempFile(ui.Image image, String prefix) async {
  final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
  final pngBytes = byteData!.buffer.asUint8List();

  final tempDir = Directory.systemTemp;
  final tempFile = File('${tempDir.path}/temp_ocr_crop_$prefix.png');
  await tempFile.writeAsBytes(pngBytes);
  return tempFile.path;
}

Future<Uint8List?> _preprocessRegion(ui.Image croppedImage) async {
  final byteData = await croppedImage.toByteData(
    format: ui.ImageByteFormat.rawRgba,
  );
  if (byteData == null) return null;
  final pixels = byteData.buffer.asUint8List();
  final w = croppedImage.width;
  final h = croppedImage.height;

  final grays = Uint8List(w * h);
  for (int i = 0; i < pixels.length; i += 4) {
    int r = pixels[i];
    int g = pixels[i + 1];
    int b = pixels[i + 2];
    grays[i ~/ 4] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0, 255);
  }

  final integral = Int32List(w * h);
  for (int y = 0; y < h; y++) {
    int rowSum = 0;
    for (int x = 0; x < w; x++) {
      rowSum += grays[y * w + x];
      if (y == 0) {
        integral[y * w + x] = rowSum;
      } else {
        integral[y * w + x] = integral[(y - 1) * w + x] + rowSum;
      }
    }
  }

  final binarized = Uint8List(pixels.length);
  final int s = (w / 8).round().clamp(7, 100);
  const double t = 0.15;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int x1 = (x - s ~/ 2).clamp(0, w - 1);
      int x2 = (x + s ~/ 2).clamp(0, w - 1);
      int y1 = (y - s ~/ 2).clamp(0, h - 1);
      int y2 = (y + s ~/ 2).clamp(0, h - 1);

      int count = (x2 - x1 + 1) * (y2 - y1 + 1);

      int sum = integral[y2 * w + x2];
      if (x1 > 0) {
        sum -= integral[y2 * w + (x1 - 1)];
      }
      if (y1 > 0) {
        sum -= integral[(y1 - 1) * w + x2];
      }
      if (x1 > 0 && y1 > 0) {
        sum += integral[(y1 - 1) * w + (x1 - 1)];
      }

      int gray = grays[y * w + x];
      int color = (gray * count < sum * (1.0 - t)) ? 0 : 255;

      int idx = (y * w + x) * 4;
      binarized[idx] = color;
      binarized[idx + 1] = color;
      binarized[idx + 2] = color;
      binarized[idx + 3] = 255;
    }
  }

  // Double dilation to fill gaps in 7-segments font
  final dilated1 = _dilate(binarized, w, h);
  final dilated2 = _dilate(dilated1, w, h);
  return dilated2;
}

Uint8List _dilate(Uint8List source, int width, int height) {
  final output = Uint8List(source.length);
  output.fillRange(0, output.length, 255);
  for (int i = 3; i < output.length; i += 4) {
    output[i] = 255;
  }

  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      bool hasForeground = false;
      for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
          int idx = ((y + ky) * width + (x + kx)) * 4;
          if (source[idx] == 0) {
            hasForeground = true;
            break;
          }
        }
        if (hasForeground) break;
      }

      int outIdx = (y * width + x) * 4;
      if (hasForeground) {
        output[outIdx] = 0;
        output[outIdx + 1] = 0;
        output[outIdx + 2] = 0;
      }
    }
  }
  return output;
}

Future<String> _saveRgbaBytesToTempFile(
  Uint8List pixels,
  int width,
  int height,
  String prefix,
) async {
  final completer = Completer<ui.Image>();
  ui.decodeImageFromPixels(
    pixels,
    width,
    height,
    ui.PixelFormat.rgba8888,
    completer.complete,
  );
  final image = await completer.future;
  return await _saveImageToTempFile(image, prefix);
}

int? _extractCleanNumber(String rawText) {
  final lines = rawText.split('\n');
  final List<int> candidates = [];

  for (var line in lines) {
    String mapped = line.toUpperCase().trim();

    mapped = mapped.replaceAll(RegExp(r'[ILi|!\\\\\\\\[\\\\\\\\]]'), '1');
    mapped = mapped.replaceAll(RegExp(r'[OoD]'), '0');
    mapped = mapped.replaceAll(RegExp(r'[S]'), '5');
    mapped = mapped.replaceAll(RegExp(r'[B]'), '8');
    mapped = mapped.replaceAll(RegExp(r'[Z]'), '2');
    mapped = mapped.replaceAll(RegExp(r'[Gg]'), '6');
    mapped = mapped.replaceAll(RegExp(r'[A]'), '4');
    mapped = mapped.replaceAll(RegExp(r'[T]'), '7');

    final matches = RegExp(r'\d+').allMatches(mapped);
    for (final match in matches) {
      final val = int.tryParse(match.group(0)!);
      if (val != null && val >= 30 && val <= 250) {
        candidates.add(val);
      }
    }
  }

  if (candidates.isEmpty) return null;
  return candidates.first;
}

String runTesseract(String imagePath, {required String model, required String psm, required bool whitelist}) {
  final args = [
    imagePath,
    'stdout',
  ];
  if (model != 'eng') {
    args.addAll(['--tessdata-dir', 'assets/tessdata', '-l', model]);
  } else {
    args.addAll(['-l', 'eng']);
  }
  if (whitelist) {
    args.addAll(['-c', 'tessedit_char_whitelist=0123456789']);
  }
  args.addAll(['--psm', psm]);
  
  final res = Process.runSync('tesseract', args);
  if (res.exitCode != 0) {
    return 'TESSERACT_ERROR: ${res.stderr}';
  }
  return res.stdout.toString().trim();
}

int runTesseractCombined(String rawPath, String? binPath, String baseModel) {
  // 1. Try raw with PSM 8
  if (File(rawPath).existsSync()) {
    String txt = runTesseract(rawPath, model: baseModel, psm: '8', whitelist: true);
    int val = _extractCleanNumber(txt) ?? 0;
    if (val >= 30 && val <= 250) return val;
  }
  // 2. Try bin with PSM 8
  if (binPath != null && File(binPath).existsSync()) {
    String txt = runTesseract(binPath, model: baseModel, psm: '8', whitelist: true);
    int val = _extractCleanNumber(txt) ?? 0;
    if (val >= 30 && val <= 250) return val;
  }
  // 3. Try raw with PSM 7
  if (File(rawPath).existsSync()) {
    String txt = runTesseract(rawPath, model: baseModel, psm: '7', whitelist: true);
    int val = _extractCleanNumber(txt) ?? 0;
    if (val >= 30 && val <= 250) return val;
  }
  // 4. Try bin with PSM 7
  if (binPath != null && File(binPath).existsSync()) {
    String txt = runTesseract(binPath, model: baseModel, psm: '7', whitelist: true);
    int val = _extractCleanNumber(txt) ?? 0;
    if (val >= 30 && val <= 250) return val;
  }
  // 5. Try raw with PSM 6
  if (File(rawPath).existsSync()) {
    String txt = runTesseract(rawPath, model: baseModel, psm: '6', whitelist: true);
    int val = _extractCleanNumber(txt) ?? 0;
    if (val >= 30 && val <= 250) return val;
  }
  // 6. Try bin with PSM 6
  if (binPath != null && File(binPath).existsSync()) {
    String txt = runTesseract(binPath, model: baseModel, psm: '6', whitelist: true);
    int val = _extractCleanNumber(txt) ?? 0;
    if (val >= 30 && val <= 250) return val;
  }
  return 0;
}

void main() {
  testWidgets('Evaluar precision de Tesseract offline', (tester) async {
    TestWidgetsFlutterBinding.ensureInitialized();

    await tester.runAsync(() async {
      final gtFile = File('test/ground_truths.json');
      if (!await gtFile.exists()) {
        fail('El archivo test/ground_truths.json no existe.');
      }
      final Map<String, dynamic> groundTruths = jsonDecode(await gtFile.readAsString());

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

      final cropsDir = Directory('test/crops');
      if (await cropsDir.exists()) {
        await cropsDir.delete(recursive: true);
      }
      await cropsDir.create(recursive: true);

      print('Cargando YOLOv8 TFLite desde archivo...');
      final interpreter = Interpreter.fromFile(File('assets/models/best_float32.tflite'));

      final configs = [
        {'model': 'ssd_combined', 'pad': 0.00, 'wl': true},
        {'model': 'ssd_combined', 'pad': 0.02, 'wl': true},
        {'model': 'ssd_combined', 'pad': 0.04, 'wl': true},
        {'model': 'ssd_combined', 'pad': 0.06, 'wl': true},
        {'model': 'ssd_combined', 'pad': 0.08, 'wl': true},
        {'model': 'ssd_combined', 'pad': 0.10, 'wl': true},
        {'model': 'ssd_combined', 'pad': 0.12, 'wl': true},
        {'model': 'ssd_combined', 'pad': 0.15, 'wl': true},
      ];

      final successCountsSYS = List.filled(configs.length, 0);
      final successCountsDIA = List.filled(configs.length, 0);
      final successCountsBoth = List.filled(configs.length, 0);
      var totalEvaluated = 0;

      final resultsFile = File('tesseract_benchmark_results.txt');
      if (await resultsFile.exists()) {
        await resultsFile.delete();
      }
      final sink = resultsFile.openWrite();
      sink.writeln('======================================================');
      sink.writeln('SWEEP DE PADDING PARA TESSERACT OCR (TODAS LAS IMÁGENES)');
      sink.writeln('======================================================\n');

      final evalFiles = files;

      for (final file in evalFiles) {
        final name = file.path.split('/').last;
        if (!groundTruths.containsKey(name)) {
          continue;
        }

        totalEvaluated++;
        final gt = groundTruths[name]!;
        final int expectedSys = gt['sys'];
        final int expectedDia = gt['dia'];

        sink.writeln('------------------------------------------------------');
        sink.writeln('FOTO: $name | Esperado: SYS=$expectedSys, DIA=$expectedDia');
        sink.writeln('------------------------------------------------------');

        var bytes = await file.readAsBytes();
        
        ui.Image? originalImage;
        double widthImg = 0.0;
        double heightImg = 0.0;
        int bestSysIndex = -1;
        int bestDiaIndex = -1;
        double maxSysScore = 0.0;
        double maxDiaScore = 0.0;
        var output = List.generate(1, (i) => List.generate(6, (j) => List.filled(8400, 0.0)));

        try {
          final rawImage = img.decodeImage(bytes);
          if (rawImage == null) continue;
          final orientedImage = img.bakeOrientation(rawImage);
          widthImg = orientedImage.width.toDouble();
          heightImg = orientedImage.height.toDouble();

          final resizedImage = img.copyResize(orientedImage, width: 640, height: 640);

          final input = List.generate(1, (i) => List.generate(640, (y) {
            return List.generate(640, (x) {
              final pixel = resizedImage.getPixel(x, y);
              return [
                pixel.r.toDouble() / 255.0,
                pixel.g.toDouble() / 255.0,
                pixel.b.toDouble() / 255.0,
              ];
            });
          }));

          interpreter.run(input, output);

          for (int i = 0; i < 8400; i++) {
            double sysScore = output[0][5][i];
            double diaScore = output[0][4][i];
            if (sysScore > maxSysScore) {
              maxSysScore = sysScore;
              bestSysIndex = i;
            }
            if (diaScore > maxDiaScore) {
              maxDiaScore = diaScore;
              bestDiaIndex = i;
            }
          }

          final orientedBytes = Uint8List.fromList(img.encodeJpg(orientedImage, quality: 90));
          final codec = await ui.instantiateImageCodec(orientedBytes);
          final frame = await codec.getNextFrame();
          originalImage = frame.image;
        } catch (e) {
          sink.writeln('  Error YOLOv8/load: $e');
          continue;
        }

        // Probar todas las configuraciones (paddings)
        for (int c = 0; c < configs.length; c++) {
          final conf = configs[c];
          final String model = conf['model'] as String;
          final double padPct = (conf['pad'] as num).toDouble();
          final bool wl = conf['wl'] as bool;

          ui.Image? sysCrop;
          ui.Image? diaCrop;

          // Crop SYS
          if (bestSysIndex != -1 && maxSysScore > 0.10 && originalImage != null) {
            double xc = output[0][0][bestSysIndex];
            double yc = output[0][1][bestSysIndex];
            double w = output[0][2][bestSysIndex];
            double h = output[0][3][bestSysIndex];

            double left = (xc - w / 2) * widthImg;
            double top = (yc - h / 2) * heightImg;
            double width = w * widthImg;
            double height = h * heightImg;

            final double padW = width * padPct;
            final double padH = height * padPct;
            final sysCropRect = ui.Rect.fromLTWH(
              (left - padW / 2).clamp(0.0, widthImg - 10),
              (top - padH / 2).clamp(0.0, heightImg - 10),
              (width + padW).clamp(10.0, widthImg - left),
              (height + padH).clamp(10.0, heightImg - top),
            );
            sysCrop = await _cropImage(originalImage, sysCropRect);
          }

          // Crop DIA
          if (bestDiaIndex != -1 && maxDiaScore > 0.10 && originalImage != null) {
            double xc = output[0][0][bestDiaIndex];
            double yc = output[0][1][bestDiaIndex];
            double w = output[0][2][bestDiaIndex];
            double h = output[0][3][bestDiaIndex];

            double left = (xc - w / 2) * widthImg;
            double top = (yc - h / 2) * heightImg;
            double width = w * widthImg;
            double height = h * heightImg;

            final double padW = width * padPct;
            final double padH = height * padPct;
            final diaCropRect = ui.Rect.fromLTWH(
              (left - padW / 2).clamp(0.0, widthImg - 10),
              (top - padH / 2).clamp(0.0, heightImg - 10),
              (width + padW).clamp(10.0, widthImg - left),
              (height + padH).clamp(10.0, heightImg - top),
            );
            diaCrop = await _cropImage(originalImage, diaCropRect);
          }

          // Guardar archivos temporales
          String? sysRawPath;
          String? diaRawPath;
          String? sysBinPath;
          String? diaBinPath;

          if (sysCrop != null) {
            sysRawPath = await _saveImageToTempFile(sysCrop, 'sys_raw_c$c');
            final binBytes = await _preprocessRegion(sysCrop);
            if (binBytes != null) {
              sysBinPath = await _saveRgbaBytesToTempFile(binBytes, sysCrop.width, sysCrop.height, 'sys_bin_c$c');
            }
          }
          if (diaCrop != null) {
            diaRawPath = await _saveImageToTempFile(diaCrop, 'dia_raw_c$c');
            final binBytes = await _preprocessRegion(diaCrop);
            if (binBytes != null) {
              diaBinPath = await _saveRgbaBytesToTempFile(binBytes, diaCrop.width, diaCrop.height, 'dia_bin_c$c');
            }
          }

          int sysVal = 0;
          int diaVal = 0;
          String rawSysText = '';
          String rawDiaText = '';

          if (sysRawPath != null) {
            sysVal = runTesseractCombined(sysRawPath, sysBinPath, 'ssd');
            rawSysText = 'COMBINED($sysVal)';
          }
          if (diaRawPath != null) {
            diaVal = runTesseractCombined(diaRawPath, diaBinPath, 'ssd');
            rawDiaText = 'COMBINED($diaVal)';
          }

          final bool sysOk = sysVal == expectedSys;
          final bool diaOk = diaVal == expectedDia;
          final bool bothOk = sysOk && diaOk;

          if (sysOk) successCountsSYS[c]++;
          if (diaOk) successCountsDIA[c]++;
          if (bothOk) successCountsBoth[c]++;

          sink.writeln('  Conf $c [Pad: ${padPct.toStringAsFixed(2)}]:');
          sink.writeln('    SYS: Extraído: $sysVal (${sysOk ? 'OK' : 'FAIL'})');
          sink.writeln('    DIA: Extraído: $diaVal (${diaOk ? 'OK' : 'FAIL'})');

          // Limpiar archivos temporales
          for (final p in [sysRawPath, diaRawPath, sysBinPath, diaBinPath]) {
            if (p != null) {
              try {
                final f = File(p);
                if (await f.exists()) await f.delete();
              } catch (_) {}
            }
          }
        }
        sink.writeln('');
      }

      interpreter.close();

      sink.writeln('======================================================');
      sink.writeln('RESUMEN DE ACCURACY');
      sink.writeln('Total evaluado: $totalEvaluated fotos');
      sink.writeln('======================================================');

      for (int c = 0; c < configs.length; c++) {
        final conf = configs[c];
        final pad = conf['pad'];
        final sysAcc = (successCountsSYS[c] / totalEvaluated * 100).toStringAsFixed(1);
        final diaAcc = (successCountsDIA[c] / totalEvaluated * 100).toStringAsFixed(1);
        final bothAcc = (successCountsBoth[c] / totalEvaluated * 100).toStringAsFixed(1);

        final summaryLine = 'Conf $c [Pad: ${pad.toString()}] => SYS: $sysAcc% (${successCountsSYS[c]}/$totalEvaluated), DIA: $diaAcc% (${successCountsDIA[c]}/$totalEvaluated), AMBOS: $bothAcc% (${successCountsBoth[c]}/$totalEvaluated)';
        sink.writeln(summaryLine);
        print(summaryLine);
      }

      sink.writeln('======================================================');
      await sink.close();
      print('Benchmark completado.');
    });
  });
}
