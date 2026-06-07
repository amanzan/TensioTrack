# TensioTrack

**Aplicación multiplataforma para el registro inteligente de la tensión arterial**

TensioTrack es un prototipo académico desarrollado con Flutter para Android, iOS y Web. Su objetivo es facilitar el registro, almacenamiento y consulta de mediciones de tensión arterial mediante una interfaz sencilla y accesible.

Se trata del Trabajo de Fin de Grado del estudiante Alberto Manzano Torregrosa para el Curso de Adaptación al Grado en Ingeniería en Informática de la universidad UNIR.

## Estado actual del proyecto

La aplicación incluye una base funcional multiplataforma con las siguientes características:

- **Pantalla de inicio** centrada en la última medición registrada.
- **Registro manual** de presión sistólica y diastólica.
- **Reconocimiento inteligente (OCR)** de valores SYS, DIA y pulso a partir de fotografías usando proveedores cloud multimodales: **Gemini Vision**, **GitHub Models** y **Groq Llama Vision**, con respaldo local/offline.
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

## Configuración del OCR cloud

Para que el reconocimiento automático por imagen funcione en modo cloud, configura al menos una clave. El orden normal de uso en móvil es Gemini, GitHub Models, Gemini, GitHub Models, Groq, Groq y después respaldo offline.

1. Obtén una API Key de Gemini en [Google AI Studio](https://aistudio.google.com/).
2. Opcionalmente, obtén una API key de Groq en [GroqCloud](https://console.groq.com/keys).
3. Para GitHub Models, crea un Personal Access Token:
   - En GitHub, ve a Settings > Developer settings > Personal access tokens.
   - Crea un fine-grained token con permiso **Models: Read** (`models:read`). No necesita permisos de repositorio.
   - Si usas un classic token, usa el scope **models**.
4. Crea un archivo llamado `.env.json` en la raíz del proyecto con esta estructura (nunca lo añadas al repositorio Git, ya está incluido en `.gitignore`):
   ```json
   {
     "GEMINI_API_KEY": "TU_API_KEY_AQUI",
     "GROQ_API_KEY": "TU_API_KEY_DE_GROQ_OPCIONAL",
     "GITHUB_MODELS_TOKEN": "TU_PAT_DE_GITHUB_MODELS",
     "GITHUB_MODELS_MODEL": "openai/gpt-4o-mini",
     "FORCE_OFFLINE_OCR": false,
     "FORCE_GITHUB_OCR": false,
     "FORCE_GROQ_OCR": false
   }
   ```

Para probar un motor concreto, cambia uno de los flags `FORCE_OFFLINE_OCR`, `FORCE_GITHUB_OCR` o `FORCE_GROQ_OCR` a `true`. El modelo por defecto de GitHub Models es `openai/gpt-4o-mini`; puedes sustituirlo por otro modelo multimodal del catálogo de GitHub Models siempre que acepte imágenes.

## Ejecución

Para instalar dependencias y ejecutar el proyecto:

```bash
flutter pub get
```

Ejecutar en móvil (Android/iOS) cargando las claves cloud:
```bash
flutter run --dart-define-from-file=.env.json
```

Para ejecutar en web con las claves cloud:
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
