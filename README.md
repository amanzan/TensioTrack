# TensioTrack

**Aplicación multiplataforma para el registro inteligente de la tensión arterial**

TensioTrack es un prototipo académico desarrollado con Flutter para Android, iOS y Web. Su objetivo es facilitar el registro, almacenamiento y consulta de mediciones de tensión arterial mediante una interfaz sencilla y accesible.

Se trata del Trabajo de Fin de Grado del estudiante Alberto Manzano Torregrosa para el Curso de Adaptación al Grado en Ingeniería en Informática de la universidad UNIR.

## Estado actual del proyecto

La aplicación incluye una base funcional multiplataforma con las siguientes características:

- **Pantalla de inicio** centrada en la última medición registrada.
- **Registro manual** de presión sistólica y diastólica.
- **Reconocimiento inteligente (OCR)** de valores SYS, DIA y pulso a partir de fotografías usando el modelo **Gemini Vision (`gemini-2.5-flash`)**.
- **Histórico de mediciones** almacenadas localmente.
- **Pantalla de estadísticas** con evolución temporal, promedios y distribución de estados.
- **Gestión de recordatorios** dentro de la aplicación.
- **Persistencia local** mediante `shared_preferences`.
- **Validación básica de valores** antes de guardar una medición.

## Funcionalidades pendientes y mejoras futuras

Tras la integración del motor OCR en la nube mediante Gemini, quedan las siguientes mejoras y líneas de desarrollo:

- **OCR local y offline**: Como alternativa al reconocimiento en la nube con Gemini, se contempla implementar reconocimiento local mediante **Tesseract** (`flutter_tesseract_ocr`) para evitar la dependencia de internet y asegurar la privacidad total de los datos.
- **Preprocesamiento avanzado de imágenes** (filtros de contraste, umbralizado) para mejorar la precisión del OCR offline.
- **Notificaciones locales reales** para avisar al usuario de los recordatorios.
- **Cifrado** o almacenamiento seguro reforzado.
- **Sincronización en la nube** y exportación de informes en PDF/CSV.

## Configuración de Gemini OCR

Para que el reconocimiento automático por imagen funcione, se requiere configurar una API Key de Gemini:

1. Obtén una API Key gratuita en [Google AI Studio](https://aistudio.google.com/).
2. Crea un archivo llamado `.env.json` en la raíz del proyecto con la siguiente estructura (nunca lo añadas al repositorio Git, ya está incluido en `.gitignore`):
   ```json
   {
     "GEMINI_API_KEY": "TU_API_KEY_AQUI"
   }
   ```

## Ejecución

Para instalar dependencias y ejecutar el proyecto:

```bash
flutter pub get
```

Ejecutar en móvil (Android/iOS) cargando la API key de Gemini:
```bash
flutter run --dart-define-from-file=.env.json
```

Para ejecutar en web con la clave de Gemini:
```bash
flutter run -d chrome --dart-define-from-file=.env.json
```

## Validación técnica actual

El proyecto se ha comprobado con:

```bash
flutter analyze
flutter test
```

Ambos comandos deben ejecutarse correctamente antes de considerar estable una modificación.
