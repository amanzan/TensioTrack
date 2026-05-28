import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'ocr_service.dart';

/// Retorna la instancia de servicio OCR web.
OcrService getOcrService() => WebOcrService();

class WebOcrService implements OcrService {
  @override
  Future<OcrResult?> recognizePressure(String imagePath, Uint8List imageBytes) async {
    // Simular un tiempo de procesamiento para el motor de reconocimiento
    await Future<void>.delayed(const Duration(milliseconds: 1400));

    // 1. Obtener las dimensiones reales de la imagen usando el decodificador nativo
    double imageWidth = 800;
    double imageHeight = 600;
    try {
      final codec = await instantiateImageCodec(imageBytes);
      final frameInfo = await codec.getNextFrame();
      imageWidth = frameInfo.image.width.toDouble();
      imageHeight = frameInfo.image.height.toDouble();
    } catch (_) {
      // Fallback si falla la lectura de dimensiones
    }

    // 2. Generar valores deterministas usando la longitud de bytes como semilla
    // Esto asegura que la misma imagen siempre de el mismo resultado, simulando un OCR real.
    final seed = imageBytes.length;
    final rand = Random(seed);

    // Valores comunes y clínicamente lógicos de tensión arterial
    final systolic = 110 + rand.nextInt(35); // Rango: 110 a 145
    final diastolic = 70 + rand.nextInt(20);  // Rango: 70 a 90

    // 3. Crear coordenadas proporcionales para dibujar los rectángulos delimitadores
    // Simulamos que la tensión sistólica está arriba en el centro y la diastólica abajo en el centro
    final sysWidth = imageWidth * 0.28;
    final sysHeight = imageHeight * 0.16;
    final sysLeft = (imageWidth - sysWidth) / 2;
    final sysTop = imageHeight * 0.28;

    final diaWidth = imageWidth * 0.28;
    final diaHeight = imageHeight * 0.16;
    final diaLeft = (imageWidth - diaWidth) / 2;
    final diaTop = imageHeight * 0.52;

    final systolicBox = Rect.fromLTWH(sysLeft, sysTop, sysWidth, sysHeight);
    final diastolicBox = Rect.fromLTWH(diaLeft, diaTop, diaWidth, diaHeight);

    return OcrResult(
      systolic: systolic,
      diastolic: diastolic,
      systolicBox: systolicBox,
      diastolicBox: diastolicBox,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      confidence: 0.85,
      engineName: 'Web OCR Engine (Simulado)',
    );
  }
}
