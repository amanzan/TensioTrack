# TensioTrack

**Aplicación multiplataforma para el registro inteligente de la tensión arterial**

TensioTrack es un prototipo académico desarrollado con Flutter para Android, iOS y Web. Su objetivo es facilitar el registro, almacenamiento y consulta de mediciones de tensión arterial mediante una interfaz sencilla y accesible.

Se trata del Trabajo de Fin de Grado del estudiante Alberto Manzano Torregrosa para el Curso de Adaptación al Grado en Ingeniería en Informática de la universidad UNIR.

## Estado actual del proyecto

La aplicación ya incluye una base funcional multiplataforma con las siguientes características:

- Pantalla de inicio centrada en la última medición registrada.
- Registro manual de presión sistólica y diastólica.
- Captura de imagen desde cámara o galería como paso previo a la futura integración OCR.
- Histórico de mediciones almacenadas localmente.
- Pantalla de estadísticas con evolución temporal, promedios y distribución de estados.
- Gestión básica de recordatorios dentro de la aplicación.
- Persistencia local mediante `shared_preferences`.
- Validación básica de valores antes de guardar una medición.

## Funcionalidades pendientes

El OCR todavía no está integrado. Actualmente, el flujo de captura permite obtener o seleccionar una imagen del tensiómetro, pero los valores deben confirmarse manualmente antes de guardarse.

También quedan como posibles mejoras futuras:

- Reconocimiento automático de valores mediante OCR.
- Preprocesamiento avanzado de imágenes.
- Notificaciones locales reales para recordatorios.
- Cifrado o almacenamiento seguro reforzado.
- Sincronización en la nube.
- Exportación de datos o informes.

## Plataformas objetivo

El proyecto está configurado para ejecutarse en:

- Android
- iOS
- Web

Al estar desarrollado en Flutter, la interfaz principal se comparte desde una única base de código ubicada en `lib/main.dart`.

## Ejecución

Para instalar dependencias y ejecutar el proyecto:

```bash
flutter pub get
flutter run
```

Para ejecutar en web:

```bash
flutter run -d chrome
```

## Validación técnica actual

El proyecto se ha comprobado con:

```bash
flutter analyze
flutter test
```

Ambos comandos deben ejecutarse correctamente antes de considerar estable una modificación.
