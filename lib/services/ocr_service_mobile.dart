import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'ocr_service.dart';

/// Retorna la instancia de servicio OCR móvil.
OcrService getOcrService() => MobileOcrService();

class _Candidate {
  _Candidate({
    required this.value,
    required this.rect,
    required this.height,
  });

  final int value;
  final Rect rect;
  final double height;
}

class MobileOcrService implements OcrService {
  @override
  Future<OcrResult?> recognizePressure(String imagePath, Uint8List imageBytes) async {
    if (imagePath.isEmpty) return null;

    final file = File(imagePath);
    if (!await file.exists()) return null;

    // 1. Obtener dimensiones reales de la imagen usando el decodificador de Flutter
    double imageWidth = 1000;
    double imageHeight = 1000;
    try {
      final codec = await instantiateImageCodec(imageBytes);
      final frameInfo = await codec.getNextFrame();
      imageWidth = frameInfo.image.width.toDouble();
      imageHeight = frameInfo.image.height.toDouble();
    } catch (_) {
      // Fallback a dimensiones estimadas si falla
    }

    final inputImage = InputImage.fromFilePath(imagePath);
    final textRecognizer = TextRecognizer(script: TextRecognitionScript.latin);

    try {
      final recognizedText = await textRecognizer.processImage(inputImage);
      final candidates = <_Candidate>[];

      // 2. Extraer todos los elementos de texto que sean números válidos en rango médico
      for (final block in recognizedText.blocks) {
        for (final line in block.lines) {
          for (final element in line.elements) {
            final text = element.text.trim();
            // Mantener solo dígitos
            final cleanText = text.replaceAll(RegExp(r'\D'), '');
            if (cleanText.isEmpty) continue;

            final value = int.tryParse(cleanText);
            if (value == null) continue;

            // Rango de presión arterial fisiológica razonable (SYS y DIA)
            if (value >= 40 && value <= 250) {
              candidates.add(_Candidate(
                value: value,
                rect: element.boundingBox,
                height: element.boundingBox.height,
              ));
            }
          }
        }
      }

      // 3. Filtrar solapamientos (evitar leer el mismo número dos veces)
      final uniqueCandidates = <_Candidate>[];
      for (final c in candidates) {
        bool isDuplicate = false;
        for (final u in uniqueCandidates) {
          // Calcular intersección
          final left = max(c.rect.left, u.rect.left);
          final top = max(c.rect.top, u.rect.top);
          final right = min(c.rect.right, u.rect.right);
          final bottom = min(c.rect.bottom, u.rect.bottom);

          if (left < right && top < bottom) {
            final overlapArea = (right - left) * (bottom - top);
            final areaC = c.rect.width * c.rect.height;
            final areaU = u.rect.width * u.rect.height;
            final minArea = min(areaC, areaU);

            if (minArea > 0 && (overlapArea / minArea) > 0.5) {
              isDuplicate = true;
              break;
            }
          }
        }
        if (!isDuplicate) {
          uniqueCandidates.add(c);
        }
      }

      // 4. Ordenar candidatos por su altura vertical de mayor a menor (tamaño visual de fuente)
      uniqueCandidates.sort((a, b) => b.height.compareTo(a.height));

      // Necesitamos al menos los dos números más grandes
      if (uniqueCandidates.length < 2) {
        return null;
      }

      // Los dos números con mayor altura visual
      final first = uniqueCandidates[0];
      final second = uniqueCandidates[1];

      // Por lógica médica, la presión sistólica (SYS) es siempre mayor que la diastólica (DIA)
      final val1 = first.value;
      final val2 = second.value;

      final systolic = val1 > val2 ? val1 : val2;
      final diastolic = val1 > val2 ? val2 : val1;

      final systolicBox = val1 > val2 ? first.rect : second.rect;
      final diastolicBox = val1 > val2 ? second.rect : first.rect;

      return OcrResult(
        systolic: systolic,
        diastolic: diastolic,
        systolicBox: systolicBox,
        diastolicBox: diastolicBox,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
        confidence: 0.92, // Alta confianza por motor nativo
        engineName: 'Google ML Kit (Nativo)',
      );
    } catch (e) {
      // Capturar cualquier error del motor nativo
      return null;
    } finally {
      await textRecognizer.close();
    }
  }
}
