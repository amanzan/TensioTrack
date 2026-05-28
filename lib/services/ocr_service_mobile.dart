import 'dart:async';
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:google_generative_ai/google_generative_ai.dart';

import 'ocr_service.dart';

/// Retorna la instancia de servicio OCR móvil (Gemini Vision).
OcrService getOcrService() => GeminiOcrService();

class GeminiOcrService implements OcrService {
  /// Clave API de Gemini.
  ///
  /// Se lee de la variable de entorno GEMINI_API_KEY definida en tiempo
  /// de compilación con --dart-define=GEMINI_API_KEY=tu_clave_aqui
  /// o con --dart-define-from-file=.env.json
  ///
  /// Para obtener tu clave gratuita:
  ///   1. Ve a https://aistudio.google.com/apikey
  ///   2. Inicia sesión con tu cuenta de Google
  ///   3. Haz clic en "Create API key"
  ///   4. Copia la clave
  static const _apiKey = String.fromEnvironment('GEMINI_API_KEY');

  /// Timeout máximo para la petición a Gemini (incluye subida de imagen).
  static const _timeout = Duration(seconds: 60);

  @override
  Future<OcrResult?> recognizePressure(
      String imagePath, Uint8List imageBytes) async {
    if (_apiKey.isEmpty) {
      debugPrint(
        'TensioTrack: API key de Gemini no configurada. '
        'Ejecuta con: flutter run --dart-define-from-file=.env.json',
      );
      return null;
    }

    debugPrint(
      'TensioTrack Gemini: Enviando imagen '
      '(${(imageBytes.length / 1024).toStringAsFixed(0)} KB)...',
    );

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

      // Enviar con timeout para evitar que la UI se quede colgada
      final response = await model
          .generateContent([content]).timeout(_timeout);
      final text = response.text?.trim();

      debugPrint('TensioTrack Gemini respuesta: "$text"');

      if (text == null || text.isEmpty) return null;

      // Parsear respuesta: esperamos "136,85,72"
      int sys = 0, dia = 0;
      final parts = text.split(',');

      if (parts.length >= 2) {
        sys = int.tryParse(parts[0].trim()) ?? 0;
        dia = int.tryParse(parts[1].trim()) ?? 0;
      } else {
        // Fallback: extraer todos los números del texto
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

      debugPrint('TensioTrack Gemini RESULTADO: SYS=$sys, DIA=$dia');

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
        engineName: 'Gemini Vision',
      );
    } on TimeoutException {
      debugPrint(
        'TensioTrack Gemini TIMEOUT: La petición tardó más de '
        '${_timeout.inSeconds}s. Comprueba la conexión a internet.',
      );
      return null;
    } catch (e) {
      debugPrint('TensioTrack Gemini ERROR: $e');
      return null;
    }
  }
}
