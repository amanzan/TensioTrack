import 'package:flutter_test/flutter_test.dart';
import 'package:intl/date_symbol_data_local.dart';
import 'package:intl/intl.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:tensiotrack/main.dart';

void main() {
  testWidgets('TensioTrack muestra la pantalla principal', (tester) async {
    TestWidgetsFlutterBinding.ensureInitialized();
    SharedPreferences.setMockInitialValues({});
    await initializeDateFormatting('es_ES');

    final store = TensioStore();
    await store.load();

    await tester.pumpWidget(TensioTrackApp(store: store));
    await tester.pumpAndSettle();

    expect(find.text('Inicio'), findsWidgets);
    expect(find.textContaining('Alberto'), findsWidgets);

    final now = DateTime.now();
    final expectedDate = DateFormat("d MMM, HH:mm", "es_ES").format(DateTime(now.year, now.month, now.day, 8, 30));
    expect(find.text('Última medición · $expectedDate'), findsOneWidget);
  });
}
