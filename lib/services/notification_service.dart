import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:timezone/data/latest_all.dart' as tz;
import 'package:timezone/timezone.dart' as tz;
import 'package:flutter_timezone/flutter_timezone.dart';

class NotificationService {
  NotificationService._internal();
  static final NotificationService _instance = NotificationService._internal();
  factory NotificationService() => _instance;

  final FlutterLocalNotificationsPlugin _plugin = FlutterLocalNotificationsPlugin();

  final _onNotificationClick = StreamController<String?>.broadcast();
  Stream<String?> get onNotificationClick => _onNotificationClick.stream;

  String? _initialPayload;
  String? get initialPayload => _initialPayload;

  void consumeInitialPayload() {
    _initialPayload = null;
  }

  Future<void> init() async {
    // 1. Inicializar las zonas horarias locales de forma offline
    tz.initializeTimeZones();
    try {
      final String timeZoneName = await FlutterTimezone.getLocalTimezone();
      tz.setLocalLocation(tz.getLocation(timeZoneName));
    } catch (e) {
      debugPrint('Error configurando la zona horaria: $e');
    }

    // 2. Configurar ajustes de inicialización
    const androidInit = AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosInit = DarwinInitializationSettings(
      requestAlertPermission: false,
      requestBadgePermission: false,
      requestSoundPermission: false,
    );
    const initSettings = InitializationSettings(android: androidInit, iOS: iosInit);

    // 3. Inicializar el plugin
    await _plugin.initialize(
      initSettings,
      onDidReceiveNotificationResponse: (response) {
        debugPrint('Notificación pulsada: ${response.payload}');
        _onNotificationClick.add(response.payload);
      },
    );

    // Comprobar si la app se abrió desde una notificación
    try {
      final launchDetails = await _plugin.getNotificationAppLaunchDetails();
      if (launchDetails != null && launchDetails.didNotificationLaunchApp) {
        _initialPayload = launchDetails.notificationResponse?.payload;
      }
    } catch (e) {
      debugPrint('Error al obtener launch details de notificaciones: $e');
    }

    // 4. Crear el canal predeterminado de Android (Android 8.0+)
    const androidChannel = AndroidNotificationChannel(
      'tensiotrack_reminders',
      'Recordatorios de Presión Arterial',
      description: 'Canal para avisar al usuario de sus tomas diarias de presión.',
      importance: Importance.max,
      playSound: true,
      enableVibration: true,
    );

    await _plugin
        .resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(androidChannel);
  }

  /// Solicita explícitamente los permisos en tiempo de ejecución (Android 13+ e iOS)
  Future<bool> requestPermissions() async {
    // Android
    final androidImplementation = _plugin
        .resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>();
    if (androidImplementation != null) {
      final granted = await androidImplementation.requestNotificationsPermission();
      await androidImplementation.requestExactAlarmsPermission();
      return granted ?? false;
    }

    // iOS
    final iosImplementation = _plugin
        .resolvePlatformSpecificImplementation<IOSFlutterLocalNotificationsPlugin>();
    if (iosImplementation != null) {
      final granted = await iosImplementation.requestPermissions(
        alert: true,
        badge: true,
        sound: true,
      );
      return granted ?? false;
    }

    return false;
  }

  /// Programa un recordatorio según su periodicidad
  Future<void> scheduleDailyNotification({
    required int id,
    required String title,
    required String body,
    required int hour,
    required int minute,
    required String repeatLabel,
    String? payload,
  }) async {
    // Solicitar permisos de forma proactiva
    await requestPermissions();

    const androidDetails = AndroidNotificationDetails(
      'tensiotrack_reminders',
      'Recordatorios de Presión Arterial',
      channelDescription: 'Canal para avisar al usuario de sus tomas diarias de presión.',
      importance: Importance.max,
      priority: Priority.high,
      playSound: true,
      enableVibration: true,
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    const details = NotificationDetails(android: androidDetails, iOS: iosDetails);

    final notificationPayload = payload ?? 'capture';

    if (repeatLabel == 'Cada día') {
      final scheduledDate = _nextInstanceOfTime(hour, minute);
      await _plugin.zonedSchedule(
        id,
        title,
        body,
        scheduledDate,
        details,
        payload: notificationPayload,
        androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
        uiLocalNotificationDateInterpretation:
            UILocalNotificationDateInterpretation.absoluteTime,
        matchDateTimeComponents: DateTimeComponents.time,
      );
      debugPrint('Recordatorio DIARIO programado: ID=$id, Hora=$hour:$minute');
    } else if (repeatLabel == 'Una vez') {
      final scheduledDate = _nextInstanceOfTime(hour, minute);
      await _plugin.zonedSchedule(
        id,
        title,
        body,
        scheduledDate,
        details,
        payload: notificationPayload,
        androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
        uiLocalNotificationDateInterpretation:
            UILocalNotificationDateInterpretation.absoluteTime,
        matchDateTimeComponents: null,
      );
      debugPrint('Recordatorio PUNTUAL programado: ID=$id, Hora=$hour:$minute');
    } else if (repeatLabel == 'Lun-Vie') {
      final weekdays = [1, 2, 3, 4, 5];
      for (final day in weekdays) {
        final scheduledDate = _nextInstanceOfWeekdayAndTime(day, hour, minute);
        await _plugin.zonedSchedule(
          id + day,
          title,
          body,
          scheduledDate,
          details,
          payload: notificationPayload,
          androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
          uiLocalNotificationDateInterpretation:
              UILocalNotificationDateInterpretation.absoluteTime,
          matchDateTimeComponents: DateTimeComponents.dayOfWeekAndTime,
        );
      }
      debugPrint('Recordatorio LABORAL programado: ID=$id (IDs ${id+1} a ${id+5}), Hora=$hour:$minute');
    }
  }

  tz.TZDateTime _nextInstanceOfTime(int hour, int minute) {
    final now = tz.TZDateTime.now(tz.local);
    var scheduledDate = tz.TZDateTime(
      tz.local,
      now.year,
      now.month,
      now.day,
      hour,
      minute,
    );
    if (scheduledDate.isBefore(now)) {
      scheduledDate = scheduledDate.add(const Duration(days: 1));
    }
    return scheduledDate;
  }

  tz.TZDateTime _nextInstanceOfWeekdayAndTime(int targetWeekday, int hour, int minute) {
    final now = tz.TZDateTime.now(tz.local);
    var scheduledDate = tz.TZDateTime(
      tz.local,
      now.year,
      now.month,
      now.day,
      hour,
      minute,
    );

    while (scheduledDate.weekday != targetWeekday) {
      scheduledDate = scheduledDate.add(const Duration(days: 1));
    }

    if (scheduledDate.isBefore(now)) {
      scheduledDate = scheduledDate.add(const Duration(days: 7));
    }

    return scheduledDate;
  }

  /// Cancela una notificación por su ID único
  Future<void> cancelNotification(int id) async {
    await _plugin.cancel(id);
    await _plugin.cancel(id + 1);
    await _plugin.cancel(id + 2);
    await _plugin.cancel(id + 3);
    await _plugin.cancel(id + 4);
    await _plugin.cancel(id + 5);
    debugPrint('Recordatorio cancelado con éxito: ID=$id');
  }

  /// Vacía por completo todas las notificaciones de la aplicación
  Future<void> cancelAllNotifications() async {
    await _plugin.cancelAll();
    debugPrint('Todas las notificaciones de la aplicación han sido eliminadas.');
  }
}
