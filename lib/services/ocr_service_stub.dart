import 'ocr_service.dart';

/// Retorna el servicio OCR para compilación base. Lanza una excepción si se llama directamente.
OcrService getOcrService() => throw UnsupportedError(
  'No se puede crear el servicio OCR sin las librerías específicas de la plataforma.',
);
