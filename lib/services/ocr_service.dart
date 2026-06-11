import 'dart:typed_data';
import 'dart:ui';

import 'ocr_service_stub.dart'
    if (dart.library.io) 'ocr_service_mobile.dart'
    if (dart.library.html) 'ocr_service_web.dart'
    if (dart.library.js_interop) 'ocr_service_web.dart';

enum OfflineOcrEngine {
  yolo,
  mlKit;

  String get label => switch (this) {
    OfflineOcrEngine.yolo => 'YOLO',
    OfflineOcrEngine.mlKit => 'ML Kit',
  };
}

class OfflineOcrConfig {
  static OfflineOcrEngine engine = OfflineOcrEngine.yolo;
}

/// Resultado del análisis OCR de una toma de presión.
class OcrResult {
  OcrResult({
    required this.systolic,
    required this.diastolic,
    this.systolicBox,
    this.diastolicBox,
    required this.imageWidth,
    required this.imageHeight,
    required this.confidence,
    required this.engineName,
  });

  /// Presión sistólica (máxima).
  final int systolic;

  /// Presión diastólica (mínima).
  final int diastolic;

  /// Rectángulo de la posición de la presión sistólica en la imagen original.
  final Rect? systolicBox;

  /// Rectángulo de la posición de la presión diastólica en la imagen original.
  final Rect? diastolicBox;

  /// Ancho original de la imagen procesada.
  final double imageWidth;

  /// Alto original de la imagen procesada.
  final double imageHeight;

  /// Confianza o fiabilidad estimada del reconocimiento (0.0 a 1.0).
  final double confidence;

  /// Nombre del motor OCR utilizado.
  final String engineName;
}

/// Contrato abstracto para el servicio OCR.
abstract class OcrService {
  /// Fábrica que redirige a la implementación adecuada de plataforma.
  factory OcrService() => getOcrService();

  /// Procesa una imagen dada su ruta local (mobile) o sus bytes (web/mobile).
  ///
  /// Retorna un [OcrResult] si se detectaron los valores, o null si falló.
  Future<OcrResult?> recognizePressure(String imagePath, Uint8List imageBytes);
}
