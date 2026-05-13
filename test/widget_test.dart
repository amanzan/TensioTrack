import 'package:flutter_test/flutter_test.dart';
import 'package:intl/date_symbol_data_local.dart';
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
    expect(find.text('Última medición · 13 may, 08:30'), findsOneWidget);
  });
}
