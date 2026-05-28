import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/date_symbol_data_local.dart';
import 'package:intl/intl.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';

import 'services/ocr_service.dart';

const _uuid = Uuid();

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initializeDateFormatting('es_ES');
  final store = TensioStore();
  await store.load();
  runApp(TensioTrackApp(store: store));
}

class TensioTrackApp extends StatelessWidget {
  const TensioTrackApp({super.key, required this.store});

  final TensioStore store;

  @override
  Widget build(BuildContext context) {
    const teal = Color(0xFF008D84);
    const ink = Color(0xFF202124);

    return AppScope(
      store: store,
      child: MaterialApp(
        debugShowCheckedModeBanner: false,
        title: 'TensioTrack',
        theme: ThemeData(
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(
            seedColor: teal,
            primary: teal,
            secondary: const Color(0xFF5E8CFF),
            surface: const Color(0xFFF7F8FA),
          ),
          scaffoldBackgroundColor: const Color(0xFFF5F6F8),
          fontFamily: 'Roboto',
          textTheme: const TextTheme(
            headlineLarge: TextStyle(fontWeight: FontWeight.w800, color: ink),
            headlineMedium: TextStyle(fontWeight: FontWeight.w800, color: ink),
            titleLarge: TextStyle(fontWeight: FontWeight.w800, color: ink),
            titleMedium: TextStyle(fontWeight: FontWeight.w700, color: ink),
            bodyLarge: TextStyle(color: ink),
          ),
          inputDecorationTheme: InputDecorationTheme(
            filled: true,
            fillColor: Colors.white,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(18),
              borderSide: BorderSide.none,
            ),
          ),
        ),
        home: const MainShell(),
      ),
    );
  }
}

class AppScope extends InheritedNotifier<TensioStore> {
  const AppScope({super.key, required TensioStore store, required super.child})
    : super(notifier: store);

  static TensioStore of(BuildContext context) {
    final scope = context.dependOnInheritedWidgetOfExactType<AppScope>();
    assert(scope != null, 'AppScope no encontrado');
    return scope!.notifier!;
  }
}

enum EntryMethod { manual, camera }

class Measurement {
  const Measurement({
    required this.id,
    required this.systolic,
    required this.diastolic,
    required this.createdAt,
    required this.method,
    this.notes = '',
  });

  final String id;
  final int systolic;
  final int diastolic;
  final DateTime createdAt;
  final EntryMethod method;
  final String notes;

  String get status {
    if (systolic >= 140 || diastolic >= 90) return 'Alto';
    if (systolic >= 130 || diastolic >= 80) return 'Elevado';
    return 'Normal';
  }

  Color get statusColor {
    if (status == 'Alto') return const Color(0xFFE55B5B);
    if (status == 'Elevado') return const Color(0xFFF6AA1C);
    return const Color(0xFF43B883);
  }

  Map<String, dynamic> toJson() => {
    'id': id,
    'systolic': systolic,
    'diastolic': diastolic,
    'createdAt': createdAt.toIso8601String(),
    'method': method.name,
    'notes': notes,
  };

  factory Measurement.fromJson(Map<String, dynamic> json) => Measurement(
    id: json['id'] as String,
    systolic: json['systolic'] as int,
    diastolic: json['diastolic'] as int,
    createdAt: DateTime.parse(json['createdAt'] as String),
    method: EntryMethod.values.firstWhere(
      (value) => value.name == json['method'],
      orElse: () => EntryMethod.manual,
    ),
    notes: json['notes'] as String? ?? '',
  );
}

class AppReminder {
  const AppReminder({
    required this.id,
    required this.title,
    required this.hour,
    required this.minute,
    required this.repeatLabel,
    required this.enabled,
  });

  final String id;
  final String title;
  final int hour;
  final int minute;
  final String repeatLabel;
  final bool enabled;

  String get timeText =>
      '${hour.toString().padLeft(2, '0')}:${minute.toString().padLeft(2, '0')}';

  AppReminder copyWith({bool? enabled}) => AppReminder(
    id: id,
    title: title,
    hour: hour,
    minute: minute,
    repeatLabel: repeatLabel,
    enabled: enabled ?? this.enabled,
  );

  Map<String, dynamic> toJson() => {
    'id': id,
    'title': title,
    'hour': hour,
    'minute': minute,
    'repeatLabel': repeatLabel,
    'enabled': enabled,
  };

  factory AppReminder.fromJson(Map<String, dynamic> json) => AppReminder(
    id: json['id'] as String,
    title: json['title'] as String,
    hour: json['hour'] as int,
    minute: json['minute'] as int,
    repeatLabel: json['repeatLabel'] as String,
    enabled: json['enabled'] as bool,
  );
}

class TensioStore extends ChangeNotifier {
  static const _measurementsKey = 'measurements';
  static const _remindersKey = 'reminders';

  final List<Measurement> _measurements = [];
  final List<AppReminder> _reminders = [];

  List<Measurement> get measurements => List.unmodifiable(_measurements);
  List<AppReminder> get reminders => List.unmodifiable(_reminders);

  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    final savedMeasurements = prefs.getString(_measurementsKey);
    final savedReminders = prefs.getString(_remindersKey);

    if (savedMeasurements == null) {
      _measurements.addAll(_demoMeasurements());
      await _saveMeasurements();
    } else {
      _measurements
        ..clear()
        ..addAll(
          (jsonDecode(savedMeasurements) as List<dynamic>)
              .cast<Map<String, dynamic>>()
              .map(Measurement.fromJson),
        );
    }

    if (savedReminders == null) {
      _reminders.addAll(_demoReminders());
      await _saveReminders();
    } else {
      _reminders
        ..clear()
        ..addAll(
          (jsonDecode(savedReminders) as List<dynamic>)
              .cast<Map<String, dynamic>>()
              .map(AppReminder.fromJson),
        );
    }

    _sort();
  }

  Future<void> addMeasurement(Measurement measurement) async {
    _measurements.add(measurement);
    _sort();
    await _saveMeasurements();
    notifyListeners();
  }

  Future<void> deleteMeasurement(String id) async {
    _measurements.removeWhere((measurement) => measurement.id == id);
    await _saveMeasurements();
    notifyListeners();
  }

  Future<void> addReminder(AppReminder reminder) async {
    _reminders.add(reminder);
    _reminders.sort((a, b) => a.timeText.compareTo(b.timeText));
    await _saveReminders();
    notifyListeners();
  }

  Future<void> toggleReminder(String id, bool enabled) async {
    final index = _reminders.indexWhere((reminder) => reminder.id == id);
    if (index == -1) return;
    _reminders[index] = _reminders[index].copyWith(enabled: enabled);
    await _saveReminders();
    notifyListeners();
  }

  void _sort() {
    _measurements.sort((a, b) => b.createdAt.compareTo(a.createdAt));
  }

  Future<void> _saveMeasurements() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _measurementsKey,
      jsonEncode(
        _measurements.map((measurement) => measurement.toJson()).toList(),
      ),
    );
  }

  Future<void> _saveReminders() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _remindersKey,
      jsonEncode(_reminders.map((reminder) => reminder.toJson()).toList()),
    );
  }

  List<Measurement> _demoMeasurements() {
    final now = DateTime.now();
    return [
      Measurement(
        id: _uuid.v4(),
        systolic: 118,
        diastolic: 76,
        createdAt: DateTime(now.year, now.month, now.day, 8, 30),
        method: EntryMethod.manual,
      ),
      Measurement(
        id: _uuid.v4(),
        systolic: 134,
        diastolic: 86,
        createdAt: now.subtract(const Duration(days: 1, hours: 2)),
        method: EntryMethod.camera,
        notes: 'Lectura revisada tras captura.',
      ),
      Measurement(
        id: _uuid.v4(),
        systolic: 122,
        diastolic: 80,
        createdAt: now.subtract(const Duration(days: 1, hours: 13)),
        method: EntryMethod.manual,
      ),
      Measurement(
        id: _uuid.v4(),
        systolic: 126,
        diastolic: 82,
        createdAt: now.subtract(const Duration(days: 3)),
        method: EntryMethod.manual,
      ),
      Measurement(
        id: _uuid.v4(),
        systolic: 115,
        diastolic: 74,
        createdAt: now.subtract(const Duration(days: 5)),
        method: EntryMethod.camera,
      ),
      Measurement(
        id: _uuid.v4(),
        systolic: 121,
        diastolic: 77,
        createdAt: now.subtract(const Duration(days: 6)),
        method: EntryMethod.manual,
      ),
    ];
  }

  List<AppReminder> _demoReminders() => [
    AppReminder(
      id: _uuid.v4(),
      title: 'Mañana',
      hour: 8,
      minute: 0,
      repeatLabel: 'Cada día',
      enabled: true,
    ),
    AppReminder(
      id: _uuid.v4(),
      title: 'Noche',
      hour: 21,
      minute: 0,
      repeatLabel: 'Cada día',
      enabled: true,
    ),
    AppReminder(
      id: _uuid.v4(),
      title: 'Mediodía',
      hour: 13,
      minute: 0,
      repeatLabel: 'Lun-Vie',
      enabled: false,
    ),
  ];
}

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  static const int _homeTab = 0;
  static const int _captureTab = 1;
  static const int _historyTab = 2;
  static const int _statsTab = 3;
  static const int _remindersTab = 4;

  int _index = 0;

  void _openManualEntry([
    EntryMethod method = EntryMethod.manual,
    int? systolic,
    int? diastolic,
  ]) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      backgroundColor: Colors.transparent,
      builder: (_) => ManualEntrySheet(
        method: method,
        initialSystolic: systolic,
        initialDiastolic: diastolic,
        onSaveSuccess: () {
          SuccessAlert.show(context, 'La medición se ha registrado correctamente.');
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 520),
        child: Scaffold(
          body: SafeArea(
            child: IndexedStack(
              index: _index,
              children: [
                HomeScreen(
                  onRegister: _openManualEntry,
                  onOpenHistory: () => _setTab(_historyTab),
                ),
                CaptureScreen(
                  onManualEntry: (systolic, diastolic) =>
                      _openManualEntry(EntryMethod.camera, systolic, diastolic),
                ),
                const HistoryScreen(),
                const StatsScreen(),
                const RemindersScreen(),
              ],
            ),
          ),
          bottomNavigationBar: NavigationBar(
            selectedIndex: _navigationIndex,
            onDestinationSelected: _setNavigationTab,
            labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
            destinations: const [
              NavigationDestination(
                icon: Icon(Icons.home_outlined),
                selectedIcon: Icon(Icons.home),
                label: 'Inicio',
              ),
              NavigationDestination(
                icon: Icon(Icons.photo_camera_outlined),
                selectedIcon: Icon(Icons.photo_camera),
                label: 'Captura',
              ),
              NavigationDestination(
                icon: Icon(Icons.show_chart_outlined),
                selectedIcon: Icon(Icons.show_chart),
                label: 'Stats',
              ),
              NavigationDestination(
                icon: Icon(Icons.alarm_outlined),
                selectedIcon: Icon(Icons.alarm),
                label: 'Avisos',
              ),
            ],
          ),
        ),
      ),
    );
  }

  int get _navigationIndex => switch (_index) {
    _homeTab || _historyTab => 0,
    _captureTab => 1,
    _statsTab => 2,
    _remindersTab => 3,
    _ => 0,
  };

  void _setNavigationTab(int index) {
    final tab = switch (index) {
      0 => _homeTab,
      1 => _captureTab,
      2 => _statsTab,
      3 => _remindersTab,
      _ => _homeTab,
    };
    _setTab(tab);
  }

  void _setTab(int index) {
    setState(() => _index = index);
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({
    super.key,
    required this.onRegister,
    required this.onOpenHistory,
  });

  final void Function([EntryMethod method]) onRegister;
  final VoidCallback onOpenHistory;

  @override
  Widget build(BuildContext context) {
    final store = AppScope.of(context);
    final last = store.measurements.isEmpty ? null : store.measurements.first;
    final hour = DateTime.now().hour;
    final greeting = hour < 14
        ? 'Buenos días'
        : hour < 20
        ? 'Buenas tardes'
        : 'Buenas noches';

    return ScreenFrame(
      title: 'Inicio',
      subtitle: '$greeting, Alberto',
      child: LayoutBuilder(
        builder: (context, constraints) {
          return SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(20, 30, 20, 24),
            child: ConstrainedBox(
              constraints: BoxConstraints(
                minHeight: max(0, constraints.maxHeight - 54),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  LastMeasurementCard(measurement: last),
                  const SizedBox(height: 18),
                  FilledButton.icon(
                    onPressed: onOpenHistory,
                    icon: const Icon(Icons.list_alt_outlined),
                    label: const Text('Historial'),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}

class LastMeasurementCard extends StatelessWidget {
  const LastMeasurementCard({super.key, required this.measurement});

  final Measurement? measurement;

  @override
  Widget build(BuildContext context) {
    final item = measurement;
    if (item == null) {
      return Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(28),
        ),
        child: const Text('Todavía no hay mediciones registradas.'),
      );
    }

    return Container(
      constraints: const BoxConstraints(minHeight: 410),
      padding: const EdgeInsets.fromLTRB(26, 30, 26, 28),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF07A99B), Color(0xFF00837D)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(24),
        boxShadow: const [
          BoxShadow(
            color: Color(0x24008D84),
            blurRadius: 28,
            offset: Offset(0, 18),
          ),
        ],
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          PressureValueRow(value: item.systolic, label: 'Sistólica'),
          const SizedBox(height: 10),
          Container(
            height: 1,
            margin: const EdgeInsets.symmetric(horizontal: 12),
            color: Colors.white.withValues(alpha: .22),
          ),
          const SizedBox(height: 10),
          PressureValueRow(value: item.diastolic, label: 'Diastólica'),
          const SizedBox(height: 22),
          Center(
            child: Chip(
              avatar: const Icon(
                Icons.check,
                color: Color(0xFF008D84),
                size: 20,
              ),
              label: Text(item.status),
              labelStyle: const TextStyle(
                color: Color(0xFF008D84),
                fontWeight: FontWeight.w900,
                fontSize: 16,
              ),
              backgroundColor: Colors.white,
              side: BorderSide.none,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            ),
          ),
          const SizedBox(height: 18),
          Text(
            'Última medición · ${DateFormat("d MMM, HH:mm", "es_ES").format(item.createdAt)}',
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.white, fontSize: 16),
          ),
        ],
      ),
    );
  }
}

class PressureValueRow extends StatelessWidget {
  const PressureValueRow({super.key, required this.value, required this.label});

  final int value;
  final String label;

  @override
  Widget build(BuildContext context) {
    return FittedBox(
      fit: BoxFit.scaleDown,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text(
            '$value',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 118,
              fontWeight: FontWeight.w900,
              height: .95,
            ),
          ),
          const SizedBox(width: 18),
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: Text(
              label,
              style: const TextStyle(
                color: Color(0xCCFFFFFF),
                fontSize: 24,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key, required this.onManualEntry});

  final void Function(int? systolic, int? diastolic) onManualEntry;

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  static const _apiKey = String.fromEnvironment('GEMINI_API_KEY');
  Uint8List? _imageBytes;
  bool _processing = false;
  OcrResult? _ocrResult;
  bool _ocrFailed = false;
  bool _apiKeyMissing = false;

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final image = await picker.pickImage(
      source: source,
      imageQuality: 80,
      maxWidth: 1280,
      maxHeight: 1280,
    );
    if (image == null) return;

    final bytes = await image.readAsBytes();
    setState(() {
      _imageBytes = bytes;
      _processing = true;
      _ocrResult = null;
      _ocrFailed = false;
      _apiKeyMissing = false;
    });

    if (_apiKey.isEmpty) {
      if (!mounted) return;
      setState(() {
        _ocrFailed = true;
        _apiKeyMissing = true;
        _processing = false;
      });
      return;
    }

    try {
      final ocrService = OcrService();
      final result = await ocrService.recognizePressure(image.path, bytes);
      if (!mounted) return;
      setState(() {
        _ocrResult = result;
        _ocrFailed = result == null;
      });
    } catch (e) {
      debugPrint('Error durante el OCR: $e');
      if (mounted) {
        setState(() {
          _ocrFailed = true;
        });
      }
    }
 finally {
      if (mounted) {
        setState(() => _processing = false);
      }
    }
  }

  void _loadDemoImage() async {
    setState(() {
      _processing = true;
      _ocrResult = null;
      _ocrFailed = false;
      _apiKeyMissing = false;
      // Pequeño byte array de imagen PNG 1x1 para simular el canvas de carga
      _imageBytes = base64Decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==");
    });

    await Future<void>.delayed(const Duration(milliseconds: 1200));

    if (!mounted) return;

    setState(() {
      _processing = false;
      _ocrResult = OcrResult(
        systolic: 128,
        diastolic: 84,
        systolicBox: const Rect.fromLTWH(130, 45, 240, 85),
        diastolicBox: const Rect.fromLTWH(130, 160, 240, 85),
        imageWidth: 500,
        imageHeight: 300,
        confidence: 0.96,
        engineName: 'TensioTrack OCR Demo',
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return ScreenFrame(
      title: 'Captura',
      subtitle: 'Fotografía del tensiómetro',
      child: ListView(
        padding: const EdgeInsets.fromLTRB(20, 14, 20, 24),
        children: [
          Container(
            height: 300,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(28),
              border: Border.all(color: const Color(0xFFE3E7EB)),
            ),
            clipBehavior: Clip.antiAlias,
            child: _imageBytes == null
                ? const CapturePlaceholder()
                : Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.memory(_imageBytes!, fit: BoxFit.contain),
                      if (_ocrResult != null)
                        CustomPaint(
                          painter: OcrOverlayPainter(result: _ocrResult!),
                        ),
                      if (_processing)
                        Container(
                          color: Colors.black.withValues(alpha: .35),
                          child: const Center(
                            child: CircularProgressIndicator(
                              color: Colors.white,
                            ),
                          ),
                        ),
                    ],
                  ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.photo_camera_outlined, size: 18),
                  label: const Text('Cámara', style: TextStyle(fontSize: 12.5)),
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library_outlined, size: 18),
                  label: const Text('Galería', style: TextStyle(fontSize: 12.5)),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _loadDemoImage,
                  icon: const Icon(Icons.auto_awesome, size: 18),
                  label: const Text('Demo', style: TextStyle(fontSize: 12.5)),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: const Color(0xFF008D84),
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 18),
          
          // Panel dinámico de información / estado del OCR
          if (_processing)
            const InfoPanel(
              icon: Icons.analytics_outlined,
              title: 'Analizando imagen...',
              text: 'Procesando la captura con el motor OCR inteligente para detectar las métricas de presión arterial...',
            )
          else if (_ocrFailed && _apiKeyMissing)
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFFFFF9E6),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: const Color(0x33F6AA1C), width: 1.5),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Row(
                    children: [
                      Icon(Icons.key_off_rounded, color: Color(0xFFF6AA1C), size: 28),
                      SizedBox(width: 12),
                      Text(
                        'Clave API no detectada',
                        style: TextStyle(
                          fontWeight: FontWeight.w900,
                          fontSize: 17,
                          color: Color(0xFFE08B00),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  const Text(
                    'No se ha detectado la clave API de Gemini necesaria para el reconocimiento inteligente.\n\n'
                    'Para solucionarlo desde Android Studio:\n'
                    ' 1. Abre el menú superior Run > Edit Configurations...\n'
                    ' 2. Selecciona tu configuración de Flutter (ej. main.dart).\n'
                    ' 3. En el campo "Additional arguments", añade:\n'
                    '    --dart-define-from-file=.env.json\n'
                    ' 4. Haz clic en Apply y luego detén y vuelve a arrancar la aplicación.',
                    style: TextStyle(color: Colors.black87, height: 1.4, fontSize: 13.5),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _loadDemoImage,
                          icon: const Icon(Icons.auto_awesome, color: Color(0xFF008D84)),
                          label: const Text('Probar con demo interactiva', style: TextStyle(color: Color(0xFF008D84))),
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: Color(0xFF008D84)),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            )
          else if (_ocrFailed)
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFFFDF2F2),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: const Color(0x33E55B5B), width: 1.5),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Row(
                    children: [
                      Icon(Icons.warning_amber_rounded, color: Color(0xFFE55B5B), size: 28),
                      SizedBox(width: 12),
                      Text(
                        'No se detectaron métricas',
                        style: TextStyle(
                          fontWeight: FontWeight.w900,
                          fontSize: 17,
                          color: Color(0xFFE55B5B),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  const Text(
                    'No logramos encontrar los números de presión arterial en esta foto de forma automática. Asegúrate de que:\n'
                    ' • La pantalla del tensiómetro esté bien enfocada y centrada.\n'
                    ' • Evites reflejos fuertes de luz o sombras marcadas.\n'
                    ' • Los números sean grandes y claramente visibles.',
                    style: TextStyle(color: Colors.black87, height: 1.4, fontSize: 13.5),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _loadDemoImage,
                          icon: const Icon(Icons.auto_awesome, color: Color(0xFF008D84)),
                          label: const Text('Probar con demo interactiva', style: TextStyle(color: Color(0xFF008D84))),
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: Color(0xFF008D84)),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            )
          else if (_ocrResult == null)
            const InfoPanel(
              icon: Icons.document_scanner_outlined,
              title: 'Reconocimiento OCR Inteligente',
              text: 'Saca una foto o elige una imagen de tu tensiómetro. TensioTrack detectará automáticamente los valores más grandes correspondientes a la presión sistólica y diastólica.',
            ),

          // Tarjeta de resultados detallados (solo si se detectaron valores)
          if (_ocrResult != null && !_processing) ...[
            const SizedBox(height: 18),
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFFEAF8F6),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: const Color(0x33008D84), width: 1.5),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.check_circle, color: Color(0xFF008D84), size: 24),
                      const SizedBox(width: 10),
                      const Text(
                        '¡Métricas Detectadas!',
                        style: TextStyle(
                          fontWeight: FontWeight.w900,
                          fontSize: 17,
                          color: Color(0xFF008D84),
                        ),
                      ),
                      const Spacer(),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: const Color(0xFF008D84).withValues(alpha: .1),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          '${(_ocrResult!.confidence * 100).toInt()}% conf.',
                          style: const TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF008D84),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      _detectedMetricColumn('SISTÓLICA', _ocrResult!.systolic > 0 ? '${_ocrResult!.systolic}' : '--', const Color(0xFF43B883)),
                      Container(width: 1, height: 40, color: const Color(0x22008D84)),
                      _detectedMetricColumn('DIASTÓLICA', _ocrResult!.diastolic > 0 ? '${_ocrResult!.diastolic}' : '--', const Color(0xFF5E8CFF)),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'Motor: ${_ocrResult!.engineName}',
                    style: const TextStyle(
                      fontSize: 12,
                      color: Colors.black45,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ],
              ),
            ),
          ],

          const SizedBox(height: 18),
          
          // Botón de acción principal adaptativo
          FilledButton.icon(
            onPressed: () {
              if (_ocrResult != null) {
                widget.onManualEntry(_ocrResult!.systolic, _ocrResult!.diastolic);
              } else {
                widget.onManualEntry(null, null);
              }
            },
            icon: Icon(
              _ocrResult != null ? Icons.check_circle_outline : Icons.edit_note_outlined,
            ),
            label: Text(
              _ocrResult != null
                  ? 'Confirmar y guardar lectura'
                  : 'Introducir valores manualmente',
            ),
            style: FilledButton.styleFrom(
              backgroundColor: _ocrResult != null ? const Color(0xFF008D84) : null,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(18),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _detectedMetricColumn(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.bold,
            color: Colors.black54,
            letterSpacing: 1.1,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 32,
            fontWeight: FontWeight.w900,
            color: color,
          ),
        ),
      ],
    );
  }
}

class OcrOverlayPainter extends CustomPainter {
  final OcrResult result;

  OcrOverlayPainter({required this.result});

  @override
  void paint(Canvas canvas, Size size) {
    if (result.imageWidth == 0 || result.imageHeight == 0) return;

    // Calcular escala y offset para BoxFit.contain
    final scaleX = size.width / result.imageWidth;
    final scaleY = size.height / result.imageHeight;
    final scale = min(scaleX, scaleY);

    final dispWidth = result.imageWidth * scale;
    final dispHeight = result.imageHeight * scale;

    final dx = (size.width - dispWidth) / 2;
    final dy = (size.height - dispHeight) / 2;

    final isDemo = result.engineName == 'TensioTrack OCR Demo';

    if (isDemo) {
      // Dibujar fondo de pantalla LCD simulado
      final lcdPaint = Paint()
        ..color = const Color(0xFFE2EAD9) // Tono verdoso LCD clásico retro
        ..style = PaintingStyle.fill;
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(dx, dy, dispWidth, dispHeight),
          const Radius.circular(20),
        ),
        lcdPaint,
      );

      // Dibujar cuadrícula LCD muy sutil
      final gridPaint = Paint()
        ..color = const Color(0x0F000000)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.0;
      for (double i = dx; i < dx + dispWidth; i += 20) {
        canvas.drawLine(Offset(i, dy), Offset(i, dy + dispHeight), gridPaint);
      }
      for (double j = dy; j < dy + dispHeight; j += 20) {
        canvas.drawLine(Offset(dx, j), Offset(dx + dispWidth, j), gridPaint);
      }

      // Dibujar etiqueta mmHg de LCD
      final labelPainter = TextPainter(
        text: const TextSpan(
          text: 'mmHg',
          style: TextStyle(
            color: Color(0x55000000),
            fontSize: 13,
            fontWeight: FontWeight.w900,
          ),
        ),
        textDirection: ui.TextDirection.ltr,
      )..layout();
      labelPainter.paint(canvas, Offset(dx + dispWidth - 60, dy + 25));
    }

    final paintSystolic = Paint()
      ..color = const Color(0x3343B883) // Verde semitransparente
      ..style = PaintingStyle.fill;

    final borderSystolic = Paint()
      ..color = const Color(0xFF43B883) // Verde
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final paintDiastolic = Paint()
      ..color = const Color(0x335E8CFF) // Azul semitransparente
      ..style = PaintingStyle.fill;

    final borderDiastolic = Paint()
      ..color = const Color(0xFF5E8CFF) // Azul
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    // Dibujar caja de Sistólica
    if (result.systolicBox != null) {
      final rect = _scaleRect(result.systolicBox!, scale, dx, dy);

      if (isDemo) {
        // Dibujar número digital LCD
        final numPainter = TextPainter(
          text: const TextSpan(
            text: '128',
            style: TextStyle(
              color: Color(0xFF1E351E), // Negro de LCD
              fontSize: 54,
              fontWeight: FontWeight.w900,
              fontFamily: 'monospace',
            ),
          ),
          textDirection: ui.TextDirection.ltr,
        )..layout();
        final textOffset = Offset(
          rect.left + (rect.width - numPainter.width) / 2,
          rect.top + (rect.height - numPainter.height) / 2,
        );
        numPainter.paint(canvas, textOffset);
      }

      canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(8)), paintSystolic);
      canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(8)), borderSystolic);
      _drawText(canvas, 'SYS: ${result.systolic}', rect.topLeft + const Offset(4, -20), const Color(0xFF43B883));
    }

    // Dibujar caja de Diastólica
    if (result.diastolicBox != null) {
      final rect = _scaleRect(result.diastolicBox!, scale, dx, dy);

      if (isDemo) {
        // Dibujar número digital LCD
        final numPainter = TextPainter(
          text: const TextSpan(
            text: '84',
            style: TextStyle(
              color: Color(0xFF1E351E), // Negro de LCD
              fontSize: 54,
              fontWeight: FontWeight.w900,
              fontFamily: 'monospace',
            ),
          ),
          textDirection: ui.TextDirection.ltr,
        )..layout();
        final textOffset = Offset(
          rect.left + (rect.width - numPainter.width) / 2,
          rect.top + (rect.height - numPainter.height) / 2,
        );
        numPainter.paint(canvas, textOffset);
      }

      canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(8)), paintDiastolic);
      canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(8)), borderDiastolic);
      _drawText(canvas, 'DIA: ${result.diastolic}', rect.topLeft + const Offset(4, -20), const Color(0xFF5E8CFF));
    }
  }

  Rect _scaleRect(Rect r, double scale, double dx, double dy) {
    return Rect.fromLTWH(
      dx + r.left * scale,
      dy + r.top * scale,
      r.width * scale,
      r.height * scale,
    );
  }

  void _drawText(Canvas canvas, String text, Offset offset, Color color) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
          backgroundColor: color,
        ),
      ),
      textDirection: ui.TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(canvas, offset);
  }

  @override
  bool shouldRepaint(covariant OcrOverlayPainter oldDelegate) {
    return oldDelegate.result != result;
  }
}

class CapturePlaceholder extends StatelessWidget {
  const CapturePlaceholder({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 118,
            height: 84,
            decoration: BoxDecoration(
              color: const Color(0xFFEAF8F6),
              borderRadius: BorderRadius.circular(18),
            ),
            child: const Icon(
              Icons.speed_outlined,
              size: 54,
              color: Color(0xFF008D84),
            ),
          ),
          const SizedBox(height: 18),
          const Text(
            'Coloca la pantalla del tensiómetro dentro del encuadre',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
          ),
          const SizedBox(height: 8),
          const Text(
            'Evita reflejos, baja iluminación e inclinaciones fuertes.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.black54),
          ),
        ],
      ),
    );
  }
}

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final store = AppScope.of(context);
    final grouped = <String, List<Measurement>>{};
    for (final measurement in store.measurements) {
      final key = _dateGroup(measurement.createdAt);
      grouped.putIfAbsent(key, () => []).add(measurement);
    }

    return ScreenFrame(
      title: 'Histórico',
      subtitle: '${store.measurements.length} registros guardados',
      child: grouped.isEmpty
          ? const EmptyState(message: 'No hay mediciones en el histórico.')
          : ListView(
              padding: const EdgeInsets.fromLTRB(20, 14, 20, 24),
              children: [
                for (final entry in grouped.entries) ...[
                  Padding(
                    padding: const EdgeInsets.fromLTRB(2, 14, 2, 8),
                    child: Text(
                      entry.key,
                      style: const TextStyle(
                        fontWeight: FontWeight.w800,
                        color: Colors.black54,
                      ),
                    ),
                  ),
                  for (final measurement in entry.value)
                    MeasurementTile(
                      measurement: measurement,
                      onDelete: () => store.deleteMeasurement(measurement.id),
                    ),
                ],
              ],
            ),
    );
  }

  String _dateGroup(DateTime date) {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final itemDay = DateTime(date.year, date.month, date.day);
    if (itemDay == today) return 'Hoy';
    if (itemDay == today.subtract(const Duration(days: 1))) return 'Ayer';
    return DateFormat('EEEE d MMMM', 'es_ES').format(date);
  }
}

class MeasurementTile extends StatelessWidget {
  const MeasurementTile({super.key, required this.measurement, this.onDelete});

  final Measurement measurement;
  final VoidCallback? onDelete;

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      color: Colors.white,
      margin: const EdgeInsets.only(bottom: 10),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: ListTile(
        leading: Icon(Icons.circle, color: measurement.statusColor, size: 14),
        title: Text(
          '${measurement.systolic}/${measurement.diastolic}',
          style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 20),
        ),
        subtitle: Text(
          '${DateFormat('HH:mm').format(measurement.createdAt)} · ${measurement.method == EntryMethod.camera ? 'Captura' : 'Manual'}'
          '${measurement.notes.isEmpty ? '' : ' · ${measurement.notes}'}',
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
        ),
        trailing: Wrap(
          spacing: 4,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            StatusPill(measurement: measurement),
            if (onDelete != null)
              IconButton(
                tooltip: 'Eliminar',
                icon: const Icon(Icons.delete_outline),
                onPressed: onDelete,
              ),
          ],
        ),
      ),
    );
  }
}

class StatsScreen extends StatelessWidget {
  const StatsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final measurements = AppScope.of(context).measurements;
    final latest = measurements.take(14).toList().reversed.toList();
    final systolicAvg = _avg(measurements.map((item) => item.systolic));
    final diastolicAvg = _avg(measurements.map((item) => item.diastolic));
    final elevated = measurements
        .where((item) => item.status != 'Normal')
        .length;

    return ScreenFrame(
      title: 'Estadísticas',
      subtitle: 'Evolución de presión arterial',
      child: measurements.isEmpty
          ? const EmptyState(message: 'Añade mediciones para ver estadísticas.')
          : ListView(
              padding: const EdgeInsets.fromLTRB(20, 14, 20, 24),
              children: [
                ChartCard(measurements: latest),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: MetricCard(
                        value: '$systolicAvg',
                        label: 'Prom. sistólica',
                        color: const Color(0xFF008D84),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: MetricCard(
                        value: '$diastolicAvg',
                        label: 'Prom. diastólica',
                        color: const Color(0xFF80B9B6),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                InfoPanel(
                  icon: elevated == 0
                      ? Icons.check_circle_outline
                      : Icons.priority_high_rounded,
                  title: elevated == 0
                      ? 'Tendencia estable'
                      : '$elevated mediciones requieren revisión',
                  text:
                      'Este análisis es orientativo y no sustituye la valoración de un profesional sanitario. '
                      'Consulta el histórico para revisar cada registro.',
                ),
                const SizedBox(height: 12),
                DistributionCard(measurements: measurements),
              ],
            ),
    );
  }

  int _avg(Iterable<int> values) {
    if (values.isEmpty) return 0;
    return (values.reduce((a, b) => a + b) / values.length).round();
  }
}

class ChartCard extends StatelessWidget {
  const ChartCard({super.key, required this.measurements});

  final List<Measurement> measurements;

  @override
  Widget build(BuildContext context) {
    final maxY =
        measurements.map((e) => e.systolic).fold(150, max).toDouble() + 10;
    final minY = max(
      40,
      measurements.map((e) => e.diastolic).fold(80, min) - 15,
    ).toDouble();

    return Container(
      height: 238,
      padding: const EdgeInsets.fromLTRB(18, 18, 18, 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(28),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Evolución temporal',
            style: TextStyle(fontWeight: FontWeight.w800),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: LineChart(
              LineChartData(
                minY: minY,
                maxY: maxY,
                gridData: const FlGridData(show: false),
                borderData: FlBorderData(show: false),
                titlesData: FlTitlesData(
                  topTitles: const AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                  rightTitles: const AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                  leftTitles: const AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 30,
                      interval: 1,
                      getTitlesWidget: (value, meta) {
                        final index = value.toInt();
                        if (index < 0 || index >= measurements.length) {
                          return const SizedBox.shrink();
                        }
                        return Text(
                          DateFormat(
                            'E',
                            'es_ES',
                          ).format(measurements[index].createdAt),
                          style: const TextStyle(
                            fontSize: 11,
                            color: Colors.black54,
                          ),
                        );
                      },
                    ),
                  ),
                ),
                lineBarsData: [
                  _line(
                    measurements,
                    (item) => item.systolic.toDouble(),
                    const Color(0xFF008D84),
                  ),
                  _line(
                    measurements,
                    (item) => item.diastolic.toDouble(),
                    const Color(0xFF9BC6C4),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  LineChartBarData _line(
    List<Measurement> items,
    double Function(Measurement item) valueOf,
    Color color,
  ) {
    return LineChartBarData(
      spots: [
        for (var index = 0; index < items.length; index++)
          FlSpot(index.toDouble(), valueOf(items[index])),
      ],
      color: color,
      barWidth: 4,
      dotData: const FlDotData(show: true),
      belowBarData: BarAreaData(show: false),
      isCurved: false,
    );
  }
}

class DistributionCard extends StatelessWidget {
  const DistributionCard({super.key, required this.measurements});

  final List<Measurement> measurements;

  @override
  Widget build(BuildContext context) {
    final normal = measurements.where((item) => item.status == 'Normal').length;
    final elevated = measurements
        .where((item) => item.status == 'Elevado')
        .length;
    final high = measurements.where((item) => item.status == 'Alto').length;
    final total = max(1, measurements.length);

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(28),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Distribución del histórico',
            style: TextStyle(fontWeight: FontWeight.w800),
          ),
          const SizedBox(height: 16),
          _bar('Normal', normal, total, const Color(0xFF43B883)),
          _bar('Elevado', elevated, total, const Color(0xFFF6AA1C)),
          _bar('Alto', high, total, const Color(0xFFE55B5B)),
        ],
      ),
    );
  }

  Widget _bar(String label, int value, int total, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          SizedBox(width: 70, child: Text(label)),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(99),
              child: LinearProgressIndicator(
                minHeight: 10,
                value: value / total,
                backgroundColor: const Color(0xFFECEFF2),
                color: color,
              ),
            ),
          ),
          const SizedBox(width: 10),
          SizedBox(
            width: 24,
            child: Text('$value', textAlign: TextAlign.right),
          ),
        ],
      ),
    );
  }
}

class RemindersScreen extends StatelessWidget {
  const RemindersScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final store = AppScope.of(context);

    return ScreenFrame(
      title: 'Recordatorios',
      subtitle: 'Avisos de medición',
      child: ListView(
        padding: const EdgeInsets.fromLTRB(20, 14, 20, 24),
        children: [
          for (final reminder in store.reminders)
            ReminderTile(
              reminder: reminder,
              onChanged: (enabled) =>
                  store.toggleReminder(reminder.id, enabled),
            ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            onPressed: () => _showReminderSheet(context),
            icon: const Icon(Icons.add),
            label: const Text('Añadir recordatorio'),
          ),
          const SizedBox(height: 14),
          const InfoPanel(
            icon: Icons.lock_outline,
            title: 'Datos locales',
            text:
                'Las mediciones y recordatorios se guardan en el dispositivo para mantener el prototipo funcional sin conexión.',
          ),
        ],
      ),
    );
  }

  void _showReminderSheet(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      backgroundColor: Colors.transparent,
      builder: (_) => ReminderSheet(
        onSaveSuccess: () {
          SuccessAlert.show(context, 'El recordatorio se ha guardado correctamente.');
        },
      ),
    );
  }
}

class ReminderTile extends StatelessWidget {
  const ReminderTile({
    super.key,
    required this.reminder,
    required this.onChanged,
  });

  final AppReminder reminder;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) {
    final icon = reminder.title.toLowerCase().contains('noche')
        ? Icons.nightlight_round
        : reminder.title.toLowerCase().contains('medio')
        ? Icons.schedule
        : Icons.wb_sunny_outlined;

    return Card(
      elevation: 0,
      color: Colors.white,
      margin: const EdgeInsets.only(bottom: 14),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
      child: ListTile(
        leading: Icon(
          icon,
          color: reminder.enabled ? const Color(0xFF008D84) : Colors.black26,
          size: 32,
        ),
        title: Text(
          reminder.title,
          style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 20),
        ),
        subtitle: Text('${reminder.timeText} · ${reminder.repeatLabel}'),
        trailing: Switch(value: reminder.enabled, onChanged: onChanged),
      ),
    );
  }
}

class ManualEntrySheet extends StatefulWidget {
  const ManualEntrySheet({
    super.key,
    required this.method,
    this.initialSystolic,
    this.initialDiastolic,
    this.onSaveSuccess,
  });

  final EntryMethod method;
  final int? initialSystolic;
  final int? initialDiastolic;
  final VoidCallback? onSaveSuccess;

  @override
  State<ManualEntrySheet> createState() => _ManualEntrySheetState();
}

class _ManualEntrySheetState extends State<ManualEntrySheet> {
  final _formKey = GlobalKey<FormState>();
  late final TextEditingController _systolic;
  late final TextEditingController _diastolic;
  final _notes = TextEditingController();
  DateTime _dateTime = DateTime.now();

  @override
  void initState() {
    super.initState();
    _systolic = TextEditingController(
      text: widget.initialSystolic != null && widget.initialSystolic! > 0 ? '${widget.initialSystolic}' : '',
    );
    _diastolic = TextEditingController(
      text: widget.initialDiastolic != null && widget.initialDiastolic! > 0 ? '${widget.initialDiastolic}' : '',
    );
    if (widget.method == EntryMethod.camera) {
      _notes.text = 'Lectura detectada automáticamente vía OCR.';
    }
  }

  @override
  void dispose() {
    _systolic.dispose();
    _diastolic.dispose();
    _notes.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SheetSurface(
      child: Form(
        key: _formKey,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              widget.method == EntryMethod.camera
                  ? 'Confirmar captura'
                  : 'Nueva medición',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                Expanded(child: _numberField(_systolic, 'Sistólica', '118')),
                const SizedBox(width: 12),
                Expanded(child: _numberField(_diastolic, 'Diastólica', '76')),
              ],
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _notes,
              decoration: const InputDecoration(labelText: 'Observaciones'),
              maxLines: 2,
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: _pickDateTime,
              icon: const Icon(Icons.event_outlined),
              label: Text(
                DateFormat('d MMM yyyy, HH:mm', 'es_ES').format(_dateTime),
              ),
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _save,
              icon: const Icon(Icons.save_outlined),
              label: const Text('Guardar medición'),
            ),
          ],
        ),
      ),
    );
  }

  TextFormField _numberField(
    TextEditingController controller,
    String label,
    String hint,
  ) {
    return TextFormField(
      controller: controller,
      decoration: InputDecoration(labelText: label, hintText: hint),
      keyboardType: TextInputType.number,
      validator: (value) {
        final number = int.tryParse(value ?? '');
        if (number == null) return 'Obligatorio';
        if (number < 40 || number > 260) return 'Valor no válido';
        return null;
      },
    );
  }

  Future<void> _pickDateTime() async {
    final date = await showDatePicker(
      context: context,
      initialDate: _dateTime,
      firstDate: DateTime(2020),
      lastDate: DateTime.now().add(const Duration(days: 1)),
    );
    if (date == null || !mounted) return;
    final time = await showTimePicker(
      context: context,
      initialTime: TimeOfDay.fromDateTime(_dateTime),
    );
    if (time == null) return;
    setState(() {
      _dateTime = DateTime(
        date.year,
        date.month,
        date.day,
        time.hour,
        time.minute,
      );
    });
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    await AppScope.of(context).addMeasurement(
      Measurement(
        id: _uuid.v4(),
        systolic: int.parse(_systolic.text),
        diastolic: int.parse(_diastolic.text),
        createdAt: _dateTime,
        method: widget.method,
        notes: _notes.text.trim(),
      ),
    );
    if (mounted) Navigator.of(context).pop();
    widget.onSaveSuccess?.call();
  }
}

class ReminderSheet extends StatefulWidget {
  const ReminderSheet({super.key, this.onSaveSuccess});

  final VoidCallback? onSaveSuccess;

  @override
  State<ReminderSheet> createState() => _ReminderSheetState();
}

class _ReminderSheetState extends State<ReminderSheet> {
  final _title = TextEditingController(text: 'Nueva toma');
  TimeOfDay _time = const TimeOfDay(hour: 8, minute: 0);
  String _repeat = 'Cada día';

  @override
  void dispose() {
    _title.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SheetSurface(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Añadir recordatorio',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 14),
          TextField(
            controller: _title,
            decoration: const InputDecoration(labelText: 'Nombre'),
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: _pickTime,
            icon: const Icon(Icons.schedule),
            label: Text(_time.format(context)),
          ),
          const SizedBox(height: 12),
          SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'Cada día', label: Text('Diario')),
              ButtonSegment(value: 'Lun-Vie', label: Text('Laboral')),
              ButtonSegment(value: 'Una vez', label: Text('Una vez')),
            ],
            selected: {_repeat},
            onSelectionChanged: (value) =>
                setState(() => _repeat = value.first),
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: _save,
            icon: const Icon(Icons.alarm_add_outlined),
            label: const Text('Guardar recordatorio'),
          ),
        ],
      ),
    );
  }

  Future<void> _pickTime() async {
    final picked = await showTimePicker(context: context, initialTime: _time);
    if (picked != null) setState(() => _time = picked);
  }

  Future<void> _save() async {
    final title = _title.text.trim().isEmpty
        ? 'Nueva toma'
        : _title.text.trim();
    await AppScope.of(context).addReminder(
      AppReminder(
        id: _uuid.v4(),
        title: title,
        hour: _time.hour,
        minute: _time.minute,
        repeatLabel: _repeat,
        enabled: true,
      ),
    );
    if (mounted) Navigator.of(context).pop();
    widget.onSaveSuccess?.call();
  }
}

class ScreenFrame extends StatelessWidget {
  const ScreenFrame({
    super.key,
    required this.title,
    required this.subtitle,
    required this.child,
  });

  final String title;
  final String subtitle;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          padding: const EdgeInsets.fromLTRB(20, 28, 20, 16),
          decoration: const BoxDecoration(
            color: Colors.white,
            border: Border(bottom: BorderSide(color: Color(0xFFE8EAED))),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: Theme.of(context).textTheme.headlineMedium),
              const SizedBox(height: 2),
              Text(
                subtitle,
                style: const TextStyle(color: Colors.black54, fontSize: 16),
              ),
            ],
          ),
        ),
        Expanded(child: child),
      ],
    );
  }
}

class ActionTile extends StatelessWidget {
  const ActionTile({
    super.key,
    required this.icon,
    required this.label,
    required this.color,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(22),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(22),
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(
            children: [
              Icon(icon, color: color, size: 34),
              const SizedBox(height: 12),
              FittedBox(
                child: Text(
                  label,
                  style: const TextStyle(
                    fontWeight: FontWeight.w800,
                    fontSize: 17,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class SectionTitle extends StatelessWidget {
  const SectionTitle({super.key, required this.title, this.trailing});

  final String title;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Text(
            title,
            style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 18),
          ),
        ),
        if (trailing != null) trailing!,
      ],
    );
  }
}

class MiniTrendCard extends StatelessWidget {
  const MiniTrendCard({super.key, required this.measurements});

  final List<Measurement> measurements;

  @override
  Widget build(BuildContext context) {
    if (measurements.isEmpty) {
      return const InfoPanel(
        icon: Icons.show_chart_outlined,
        title: 'Sin datos suficientes',
        text: 'Registra la primera medición para activar el seguimiento.',
      );
    }

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        children: [
          for (final item in measurements.take(3))
            MeasurementTile(measurement: item),
        ],
      ),
    );
  }
}

class MetricCard extends StatelessWidget {
  const MetricCard({
    super.key,
    required this.value,
    required this.label,
    required this.color,
  });

  final String value;
  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        children: [
          FittedBox(
            child: Text(
              value,
              style: TextStyle(
                fontSize: 42,
                fontWeight: FontWeight.w900,
                color: color,
              ),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            label,
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.black54),
          ),
        ],
      ),
    );
  }
}

class InfoPanel extends StatelessWidget {
  const InfoPanel({
    super.key,
    required this.icon,
    required this.title,
    required this.text,
  });

  final IconData icon;
  final String title;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: const Color(0xFF008D84), size: 28),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 16,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  text,
                  style: const TextStyle(color: Colors.black54, height: 1.35),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class StatusPill extends StatelessWidget {
  const StatusPill({super.key, required this.measurement});

  final Measurement measurement;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: measurement.statusColor.withValues(alpha: .13),
        borderRadius: BorderRadius.circular(99),
      ),
      child: Text(
        measurement.status,
        style: TextStyle(
          color: measurement.statusColor,
          fontWeight: FontWeight.w800,
          fontSize: 12,
        ),
      ),
    );
  }
}

class EmptyState extends StatelessWidget {
  const EmptyState({super.key, required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(28),
        child: Text(
          message,
          textAlign: TextAlign.center,
          style: const TextStyle(color: Colors.black54),
        ),
      ),
    );
  }
}

class SheetSurface extends StatelessWidget {
  const SheetSurface({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(bottom: MediaQuery.viewInsetsOf(context).bottom),
      child: Container(
        padding: const EdgeInsets.fromLTRB(20, 14, 20, 22),
        decoration: const BoxDecoration(
          color: Color(0xFFF5F6F8),
          borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 44,
                height: 5,
                margin: const EdgeInsets.only(bottom: 18),
                decoration: BoxDecoration(
                  color: const Color(0xFFD3D7DB),
                  borderRadius: BorderRadius.circular(99),
                ),
              ),
              child,
            ],
          ),
        ),
      ),
    );
  }
}

class SuccessAlert {
  static void show(BuildContext context, String message) {
    final overlayState = Overlay.of(context);
    late OverlayEntry overlayEntry;

    overlayEntry = OverlayEntry(
      builder: (context) => _SuccessAlertWidget(
        message: message,
        onDismissed: () {
          overlayEntry.remove();
        },
      ),
    );

    overlayState.insert(overlayEntry);
  }
}

class _SuccessAlertWidget extends StatefulWidget {
  const _SuccessAlertWidget({required this.message, required this.onDismissed});

  final String message;
  final VoidCallback onDismissed;

  @override
  State<_SuccessAlertWidget> createState() => _SuccessAlertWidgetState();
}

class _SuccessAlertWidgetState extends State<_SuccessAlertWidget>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _scaleAnimation;
  late final Animation<double> _opacityAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 350),
    );

    _scaleAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutBack,
    );

    _opacityAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeIn,
    );

    _controller.forward();

    // Auto dismiss after 1.8 seconds
    Future.delayed(const Duration(milliseconds: 1800), () {
      if (mounted) {
        _controller.reverse().then((_) {
          widget.onDismissed();
        });
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: Center(
        child: FadeTransition(
          opacity: _opacityAnimation,
          child: ScaleTransition(
            scale: _scaleAnimation,
            child: Container(
              width: 240,
              padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(28),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.08),
                    blurRadius: 24,
                    offset: const Offset(0, 8),
                  ),
                ],
                border: Border.all(color: const Color(0xFFE8ECEF), width: 1.5),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: const BoxDecoration(
                      color: Color(0xFFE8F8F5),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.check_circle_rounded,
                      color: Color(0xFF008D84),
                      size: 44,
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    '¡Guardado!',
                    style: TextStyle(
                      fontSize: 19,
                      fontWeight: FontWeight.w900,
                      color: Color(0xFF202124),
                      height: 1.2,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    widget.message,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      fontSize: 13,
                      color: Color(0xFF5F6368),
                      height: 1.3,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
