import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:google_generative_ai/google_generative_ai.dart';

import 'ocr_service.dart';

/// Retorna la instancia de servicio OCR web (Gemini Vision).
OcrService getOcrService() => GeminiWebOcrService();

class GeminiWebOcrService implements OcrService {
  /// Clave API de Gemini (misma configuración que en móvil).
  static const _apiKey = String.fromEnvironment('GEMINI_API_KEY');

  @override
  Future<OcrResult?> recognizePressure(
      String imagePath, Uint8List imageBytes) async {
    if (_apiKey.isEmpty) {
      debugPrint(
        'TensioTrack Web: API key de Gemini no configurada. '
        'Ejecuta con: flutter run --dart-define=GEMINI_API_KEY=tu_clave',
      );
      return null;
    }

    try {
      final model = GenerativeModel(
        model: 'gemini-2.5-flash',
        apiKey: _apiKey,
      );

      const prompt = '''Analyze this photo of a blood pressure monitor (tensiometer).
Read the numeric values shown on the LCD screen display.

The LCD screen typically shows three values:
- Systolic pressure (SYS): usually the top and largest number
- Diastolic pressure (DIA): usually below SYS, second largest
- Pulse rate (PULSE): usually the smallest number at the bottom

Respond ONLY with three integers separated by commas in this exact format:
systolic,diastolic,pulse

Example responses:
136,85,72
120,80,65

If you cannot read a value clearly, use 0 for that value.
Do NOT include any other text, explanation, units, or formatting.''';

      final content = Content.multi([
        TextPart(prompt),
        DataPart('image/jpeg', imageBytes),
      ]);

      debugPrint('TensioTrack Web Gemini: Enviando imagen...');
      final response = await model.generateContent([content]);
      final text = response.text?.trim();

      debugPrint('TensioTrack Web Gemini respuesta: "$text"');

      if (text == null || text.isEmpty) return null;

      int sys = 0, dia = 0;
      final parts = text.split(',');

      if (parts.length >= 2) {
        sys = int.tryParse(parts[0].trim()) ?? 0;
        dia = int.tryParse(parts[1].trim()) ?? 0;
      } else {
        final numbers = RegExp(r'\d+')
            .allMatches(text)
            .map((m) => int.parse(m.group(0)!))
            .where((n) => n >= 30 && n <= 250)
            .toList();

        if (numbers.length >= 2) {
          numbers.sort((a, b) => b.compareTo(a));
          sys = numbers[0];
          dia = numbers[1];
        } else if (numbers.length == 1) {
          if (numbers[0] >= 95) {
            sys = numbers[0];
          } else {
            dia = numbers[0];
          }
        }
      }

      if (sys == 0 && dia == 0) return null;

      debugPrint('TensioTrack Web Gemini RESULTADO: SYS=$sys, DIA=$dia');

      // Obtener dimensiones de la imagen
      double imageWidth = 1000, imageHeight = 1000;
      try {
        final codec = await instantiateImageCodec(imageBytes);
        final frame = await codec.getNextFrame();
        imageWidth = frame.image.width.toDouble();
        imageHeight = frame.image.height.toDouble();
      } catch (_) {}

      return OcrResult(
        systolic: sys,
        diastolic: dia,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
        confidence: (sys > 0 && dia > 0) ? 0.95 : 0.6,
        engineName: 'Gemini Vision (Web)',
      );
    } catch (e) {
      debugPrint('TensioTrack Web Gemini ERROR: $e');
      rethrow;
    }
  }
}
