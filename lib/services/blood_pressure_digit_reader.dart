import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/foundation.dart' show compute, debugPrint;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class DigitDetection {
  const DigitDetection({
    required this.digit,
    required this.confidence,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });

  final int digit;
  final double confidence;
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  double get cx => (x1 + x2) / 2.0;
  double get cy => (y1 + y2) / 2.0;
  double get width => x2 - x1;
  double get height => y2 - y1;
  Rect get rect => Rect.fromLTRB(x1, y1, x2, y2);
}

class BloodPressureReadingResult {
  const BloodPressureReadingResult({
    required this.systolic,
    required this.diastolic,
    required this.detections,
    required this.rows,
    required this.rowValues,
    required this.imageWidth,
    required this.imageHeight,
    required this.confidence,
  });

  final int systolic;
  final int diastolic;
  final List<DigitDetection> detections;
  final List<List<DigitDetection>> rows;
  final List<int> rowValues;
  final int imageWidth;
  final int imageHeight;
  final double confidence;
}

class BloodPressureDigitReader {
  BloodPressureDigitReader({
    this.modelAssetPath = 'best_float32.tflite',
    this.confidenceThreshold = 0.5,
    this.recoveryConfidenceThreshold = 0.10,
    this.iouThreshold = 0.45,
    this.rowYThresholdFactor = 0.6,
  });

  static const inputSize = 640;
  static const outputAttributeCount = 14;

  final String modelAssetPath;
  final double confidenceThreshold;
  final double recoveryConfidenceThreshold;
  final double iouThreshold;
  final double rowYThresholdFactor;

  Interpreter? _interpreter;

  Future<BloodPressureReadingResult?> readFromImagePath(
    String imagePath,
  ) async {
    final imageBytes = await File(imagePath).readAsBytes();
    return readFromImageBytes(imageBytes);
  }

  Future<BloodPressureReadingResult?> readFromImageBytes(
    Uint8List imageBytes,
  ) async {
    final preprocessed = await compute(_preprocessImageForYolo, imageBytes);
    final output = await _runInference(preprocessed.input.buffer);
    final detections = _decodeDetections(
      output,
      originalWidth: preprocessed.imageWidth,
      originalHeight: preprocessed.imageHeight,
    );
    if (detections.isEmpty) return null;

    final rows = _groupRows(detections);
    _logRows(
      rows,
      imageWidth: preprocessed.imageWidth,
      imageHeight: preprocessed.imageHeight,
    );
    if (rows.length < 2) return null;

    final rowValues = rows
        .map((row) => int.tryParse(row.map((d) => d.digit).join()))
        .whereType<int>()
        .toList(growable: false);
    if (rowValues.length < 2) return null;

    final firstTwoRows = rows.take(2).expand((row) => row);
    final confidence =
        firstTwoRows.map((d) => d.confidence).reduce((a, b) => a + b) /
        firstTwoRows.length;

    return BloodPressureReadingResult(
      systolic: rowValues[0],
      diastolic: rowValues[1],
      detections: detections,
      rows: rows,
      rowValues: rowValues,
      imageWidth: preprocessed.imageWidth,
      imageHeight: preprocessed.imageHeight,
      confidence: confidence.clamp(0.0, 1.0),
    );
  }

  Future<void> dispose() async {
    _interpreter?.close();
    _interpreter = null;
  }

  Future<Interpreter> _loadInterpreter() async {
    final current = _interpreter;
    if (current != null) return current;

    final interpreter = await Interpreter.fromAsset(modelAssetPath);
    _interpreter = interpreter;
    return interpreter;
  }

  Future<List<List<List<double>>>> _runInference(ByteBuffer input) async {
    final interpreter = await _loadInterpreter();
    final outputShape = interpreter.getOutputTensors().first.shape;
    if (outputShape.length != 3 || outputShape.first != 1) {
      throw StateError('Shape de salida TFLite inesperado: $outputShape');
    }

    final inputShape = interpreter.getInputTensors().first.shape;
    if (inputShape.length != 4 ||
        inputShape[0] != 1 ||
        inputShape[1] != inputSize ||
        inputShape[2] != inputSize ||
        inputShape[3] != 3) {
      throw StateError('Shape de entrada TFLite inesperado: $inputShape');
    }

    final output = List.generate(
      outputShape[0],
      (_) => List.generate(
        outputShape[1],
        (_) => List<double>.filled(outputShape[2], 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    interpreter.run(input, output);
    return output;
  }

  List<DigitDetection> _decodeDetections(
    List<List<List<double>>> output, {
    required int originalWidth,
    required int originalHeight,
  }) {
    final pred = output[0];
    final dim1 = pred.length;
    final dim2 = pred.first.length;
    final attributesFirst = dim1 == outputAttributeCount;
    final attributesCount = attributesFirst ? dim1 : dim2;
    final candidateCount = attributesFirst ? dim2 : dim1;

    if (attributesCount < outputAttributeCount) {
      throw StateError('Shape de salida YOLO inesperado: [1, $dim1, $dim2]');
    }

    double valueAt(int candidateIndex, int attributeIndex) {
      return attributesFirst
          ? pred[attributeIndex][candidateIndex]
          : pred[candidateIndex][attributeIndex];
    }

    final boxes = <_ModelBox>[];
    final scores = <double>[];
    final classIds = <int>[];
    final rejectedBoxes = <_ModelBox>[];
    final rejectedScores = <double>[];
    final rejectedClassIds = <int>[];

    for (var i = 0; i < candidateCount; i++) {
      final x = valueAt(i, 0);
      final y = valueAt(i, 1);
      final w = valueAt(i, 2);
      final h = valueAt(i, 3);

      var classId = 0;
      var score = valueAt(i, 4);
      for (var classIndex = 1; classIndex < 10; classIndex++) {
        final classScore = valueAt(i, 4 + classIndex);
        if (classScore > score) {
          score = classScore;
          classId = classIndex;
        }
      }

      final x1 = ((x - w / 2.0) * inputSize).toInt();
      final y1 = ((y - h / 2.0) * inputSize).toInt();
      final x2 = ((x + w / 2.0) * inputSize).toInt();
      final y2 = ((y + h / 2.0) * inputSize).toInt();
      final box = _ModelBox(x: x1, y: y1, width: x2 - x1, height: y2 - y1);

      if (score < confidenceThreshold) {
        if (score >= recoveryConfidenceThreshold) {
          rejectedBoxes.add(box);
          rejectedScores.add(score);
          rejectedClassIds.add(classId);
        }
        continue;
      }

      boxes.add(box);
      scores.add(score);
      classIds.add(classId);
    }

    _logModelDetections(
      'YOLO rechazadas pre-threshold',
      boxes: rejectedBoxes,
      scores: rejectedScores,
      classIds: rejectedClassIds,
    );

    _logModelDetections(
      'YOLO antes de NMS',
      boxes: boxes,
      scores: scores,
      classIds: classIds,
    );

    final keptIndices = _nonMaxSuppression(boxes, scores, classIds);
    final detections = <DigitDetection>[];

    for (final index in keptIndices) {
      final box = boxes[index];
      final x1 = box.x * originalWidth / inputSize;
      final y1 = box.y * originalHeight / inputSize;
      final x2 = (box.x + box.width) * originalWidth / inputSize;
      final y2 = (box.y + box.height) * originalHeight / inputSize;

      detections.add(
        DigitDetection(
          digit: classIds[index],
          confidence: scores[index],
          x1: x1.clamp(0.0, originalWidth.toDouble()),
          y1: y1.clamp(0.0, originalHeight.toDouble()),
          x2: x2.clamp(0.0, originalWidth.toDouble()),
          y2: y2.clamp(0.0, originalHeight.toDouble()),
        ),
      );
    }

    final recoveredDetections = _recoverAlignedRejectedDetections(
      detections,
      rejectedBoxes: rejectedBoxes,
      rejectedScores: rejectedScores,
      rejectedClassIds: rejectedClassIds,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
    );
    detections.addAll(recoveredDetections);
    final resolvedDetections = _resolveAmbiguousAcceptedDetections(
      detections,
      rejectedBoxes: rejectedBoxes,
      rejectedScores: rejectedScores,
      rejectedClassIds: rejectedClassIds,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
    );

    resolvedDetections.sort((a, b) {
      final yCompare = a.cy.compareTo(b.cy);
      return yCompare != 0 ? yCompare : a.cx.compareTo(b.cx);
    });

    _logImageDetections(
      'YOLO despues de NMS',
      resolvedDetections,
      imageWidth: originalWidth,
      imageHeight: originalHeight,
    );
    return resolvedDetections;
  }

  List<DigitDetection> _resolveAmbiguousAcceptedDetections(
    List<DigitDetection> detections, {
    required List<_ModelBox> rejectedBoxes,
    required List<double> rejectedScores,
    required List<int> rejectedClassIds,
    required int originalWidth,
    required int originalHeight,
  }) {
    return [
      for (final detection in detections)
        _resolveAcceptedDigitAmbiguity(
          detection,
          rejectedBoxes: rejectedBoxes,
          rejectedScores: rejectedScores,
          rejectedClassIds: rejectedClassIds,
          originalWidth: originalWidth,
          originalHeight: originalHeight,
        ),
    ];
  }

  DigitDetection _resolveAcceptedDigitAmbiguity(
    DigitDetection detection, {
    required List<_ModelBox> rejectedBoxes,
    required List<double> rejectedScores,
    required List<int> rejectedClassIds,
    required int originalWidth,
    required int originalHeight,
  }) {
    if (detection.digit != 2 || detection.confidence >= 0.70) {
      return detection;
    }

    var bestAlternativeScore = -1.0;
    for (var i = 0; i < rejectedBoxes.length; i++) {
      if (rejectedClassIds[i] != 3) continue;
      final alternative = _detectionFromModelBox(
        rejectedBoxes[i],
        digit: 3,
        confidence: rejectedScores[i],
        originalWidth: originalWidth,
        originalHeight: originalHeight,
      );
      if (_detectionIoU(detection, alternative) < 0.75) continue;
      bestAlternativeScore = math.max(bestAlternativeScore, rejectedScores[i]);
    }

    if (bestAlternativeScore >= detection.confidence - 0.15) {
      debugPrint(
        'TensioTrack YOLO ambiguedad aceptada 2/3: '
        '2=${detection.confidence.toStringAsFixed(3)} '
        '3=${bestAlternativeScore.toStringAsFixed(3)} -> usando 3',
      );
      return DigitDetection(
        digit: 3,
        confidence: bestAlternativeScore,
        x1: detection.x1,
        y1: detection.y1,
        x2: detection.x2,
        y2: detection.y2,
      );
    }

    return detection;
  }

  List<DigitDetection> _recoverAlignedRejectedDetections(
    List<DigitDetection> accepted, {
    required List<_ModelBox> rejectedBoxes,
    required List<double> rejectedScores,
    required List<int> rejectedClassIds,
    required int originalWidth,
    required int originalHeight,
  }) {
    if (accepted.isEmpty || rejectedBoxes.isEmpty) return const [];

    final recovered = <DigitDetection>[];
    final candidateOrder = List<int>.generate(rejectedBoxes.length, (i) => i)
      ..sort((a, b) => rejectedScores[b].compareTo(rejectedScores[a]));

    for (final index in candidateOrder) {
      final recoveredClass = _recoveredClassForAmbiguousCandidate(
        index,
        rejectedBoxes: rejectedBoxes,
        rejectedScores: rejectedScores,
        rejectedClassIds: rejectedClassIds,
      );
      final candidate = _detectionFromModelBox(
        rejectedBoxes[index],
        digit: recoveredClass.classId,
        confidence: recoveredClass.confidence,
        originalWidth: originalWidth,
        originalHeight: originalHeight,
      );
      final currentDetections = [...accepted, ...recovered];

      final overlapsExisting = currentDetections.any(
        (detection) => _detectionIoU(detection, candidate) > iouThreshold,
      );
      if (overlapsExisting) continue;

      final rows = _groupRows(currentDetections);
      List<DigitDetection>? matchedRow;
      for (final row in rows) {
        final rowCy = _averageCy(row);
        final avgHeight =
            row.map((d) => d.height).reduce((a, b) => a + b) / row.length;
        if ((candidate.cy - rowCy).abs() < avgHeight * rowYThresholdFactor) {
          matchedRow = row;
          break;
        }
      }
      if (matchedRow == null || matchedRow.length >= 3) continue;

      debugPrint(
        'TensioTrack YOLO recupera bajo threshold: '
        'digit=${candidate.digit} '
        'confidence=${candidate.confidence.toStringAsFixed(3)} '
        'cx=${(candidate.cx / originalWidth).toStringAsFixed(3)} '
        'cy=${(candidate.cy / originalHeight).toStringAsFixed(3)} '
        'width=${(candidate.width / originalWidth).toStringAsFixed(3)} '
        'height=${(candidate.height / originalHeight).toStringAsFixed(3)}',
      );
      recovered.add(candidate);
    }

    return recovered;
  }

  _RecoveredClass _recoveredClassForAmbiguousCandidate(
    int index, {
    required List<_ModelBox> rejectedBoxes,
    required List<double> rejectedScores,
    required List<int> rejectedClassIds,
  }) {
    final classId = rejectedClassIds[index];
    final confidence = rejectedScores[index];
    if (classId != 2 || confidence >= 0.20) {
      return _RecoveredClass(classId, confidence);
    }

    var bestAlternativeScore = -1.0;
    for (var i = 0; i < rejectedBoxes.length; i++) {
      if (i == index || rejectedClassIds[i] != 3) continue;
      final iou = _intersectionOverUnion(
        rejectedBoxes[index],
        rejectedBoxes[i],
      );
      if (iou < 0.75) continue;
      bestAlternativeScore = math.max(bestAlternativeScore, rejectedScores[i]);
    }

    if (bestAlternativeScore >= confidence - 0.05) {
      debugPrint(
        'TensioTrack YOLO ambiguedad 2/3: '
        '2=${confidence.toStringAsFixed(3)} '
        '3=${bestAlternativeScore.toStringAsFixed(3)} -> usando 3',
      );
      return _RecoveredClass(3, bestAlternativeScore);
    }

    return _RecoveredClass(classId, confidence);
  }

  DigitDetection _detectionFromModelBox(
    _ModelBox box, {
    required int digit,
    required double confidence,
    required int originalWidth,
    required int originalHeight,
  }) {
    final x1 = box.x * originalWidth / inputSize;
    final y1 = box.y * originalHeight / inputSize;
    final x2 = (box.x + box.width) * originalWidth / inputSize;
    final y2 = (box.y + box.height) * originalHeight / inputSize;

    return DigitDetection(
      digit: digit,
      confidence: confidence,
      x1: x1.clamp(0.0, originalWidth.toDouble()),
      y1: y1.clamp(0.0, originalHeight.toDouble()),
      x2: x2.clamp(0.0, originalWidth.toDouble()),
      y2: y2.clamp(0.0, originalHeight.toDouble()),
    );
  }

  double _detectionIoU(DigitDetection a, DigitDetection b) {
    final intersectionLeft = math.max(a.x1, b.x1);
    final intersectionTop = math.max(a.y1, b.y1);
    final intersectionRight = math.min(a.x2, b.x2);
    final intersectionBottom = math.min(a.y2, b.y2);
    final intersectionWidth = math.max(
      0.0,
      intersectionRight - intersectionLeft,
    );
    final intersectionHeight = math.max(
      0.0,
      intersectionBottom - intersectionTop,
    );
    final intersectionArea = intersectionWidth * intersectionHeight;
    final unionArea =
        a.width * a.height + b.width * b.height - intersectionArea;

    if (unionArea <= 0) return 0;
    return intersectionArea / unionArea;
  }

  List<int> _nonMaxSuppression(
    List<_ModelBox> boxes,
    List<double> scores,
    List<int> classIds,
  ) {
    final order = List<int>.generate(boxes.length, (i) => i)
      ..sort((a, b) => scores[b].compareTo(scores[a]));
    final kept = <int>[];

    while (order.isNotEmpty) {
      final current = order.removeAt(0);
      kept.add(current);

      order.removeWhere((candidate) {
        final iou = _intersectionOverUnion(boxes[current], boxes[candidate]);
        final shouldRemove = iou > iouThreshold;
        if (shouldRemove) {
          debugPrint(
            'TensioTrack YOLO NMS elimina: '
            'kept_digit=${classIds[current]} kept_conf=${scores[current].toStringAsFixed(3)} '
            'removed_digit=${classIds[candidate]} removed_conf=${scores[candidate].toStringAsFixed(3)} '
            'iou=${iou.toStringAsFixed(3)} '
            'kept_cx=${_norm(boxes[current].cx).toStringAsFixed(3)} '
            'removed_cx=${_norm(boxes[candidate].cx).toStringAsFixed(3)}',
          );
        }
        return shouldRemove;
      });
    }

    return kept;
  }

  double _intersectionOverUnion(_ModelBox a, _ModelBox b) {
    final ax2 = a.x + a.width;
    final ay2 = a.y + a.height;
    final bx2 = b.x + b.width;
    final by2 = b.y + b.height;

    final intersectionLeft = math.max(a.x, b.x);
    final intersectionTop = math.max(a.y, b.y);
    final intersectionRight = math.min(ax2, bx2);
    final intersectionBottom = math.min(ay2, by2);
    final intersectionWidth = math.max(0, intersectionRight - intersectionLeft);
    final intersectionHeight = math.max(
      0,
      intersectionBottom - intersectionTop,
    );
    final intersectionArea = intersectionWidth * intersectionHeight;
    final unionArea = a.area + b.area - intersectionArea;

    if (unionArea <= 0) return 0;
    return intersectionArea / unionArea;
  }

  List<List<DigitDetection>> _groupRows(List<DigitDetection> detections) {
    if (detections.isEmpty) return const [];

    final sorted = [...detections]..sort((a, b) => a.cy.compareTo(b.cy));
    final avgHeight =
        sorted.map((d) => d.height).reduce((a, b) => a + b) / sorted.length;
    final yThreshold = avgHeight * rowYThresholdFactor;
    final rows = <List<DigitDetection>>[];

    for (final detection in sorted) {
      var placed = false;

      for (final row in rows) {
        final rowCy = _averageCy(row);
        if ((detection.cy - rowCy).abs() < yThreshold) {
          row.add(detection);
          placed = true;
          break;
        }
      }

      if (!placed) {
        rows.add([detection]);
      }
    }

    for (final row in rows) {
      row.sort((a, b) => a.cx.compareTo(b.cx));
    }
    rows.sort((a, b) => _averageCy(a).compareTo(_averageCy(b)));
    return rows;
  }

  double _averageCy(List<DigitDetection> row) {
    return row.map((d) => d.cy).reduce((a, b) => a + b) / row.length;
  }

  void _logModelDetections(
    String title, {
    required List<_ModelBox> boxes,
    required List<double> scores,
    required List<int> classIds,
  }) {
    debugPrint(
      'TensioTrack $title: ${boxes.length} detecciones '
      '(conf>=${confidenceThreshold.toStringAsFixed(2)})',
    );
    for (var i = 0; i < boxes.length; i++) {
      final box = boxes[i];
      debugPrint(
        '  [$i] digit=${classIds[i]} '
        'confidence=${scores[i].toStringAsFixed(3)} '
        'cx=${_norm(box.cx).toStringAsFixed(3)} '
        'cy=${_norm(box.cy).toStringAsFixed(3)} '
        'width=${_norm(box.width).toStringAsFixed(3)} '
        'height=${_norm(box.height).toStringAsFixed(3)}',
      );
    }
  }

  void _logImageDetections(
    String title,
    List<DigitDetection> detections, {
    required int imageWidth,
    required int imageHeight,
  }) {
    debugPrint('TensioTrack $title: ${detections.length} detecciones');
    for (var i = 0; i < detections.length; i++) {
      final detection = detections[i];
      debugPrint(
        '  [$i] digit=${detection.digit} '
        'confidence=${detection.confidence.toStringAsFixed(3)} '
        'cx=${(detection.cx / imageWidth).toStringAsFixed(3)} '
        'cy=${(detection.cy / imageHeight).toStringAsFixed(3)} '
        'width=${(detection.width / imageWidth).toStringAsFixed(3)} '
        'height=${(detection.height / imageHeight).toStringAsFixed(3)}',
      );
    }
  }

  void _logRows(
    List<List<DigitDetection>> rows, {
    required int imageWidth,
    required int imageHeight,
  }) {
    debugPrint('TensioTrack YOLO filas agrupadas: ${rows.length}');
    for (var rowIndex = 0; rowIndex < rows.length; rowIndex++) {
      final row = rows[rowIndex];
      final value = row.map((d) => d.digit).join();
      debugPrint('  fila ${rowIndex + 1}: value=$value count=${row.length}');
      for (var i = 0; i < row.length; i++) {
        final detection = row[i];
        debugPrint(
          '    [$i] digit=${detection.digit} '
          'confidence=${detection.confidence.toStringAsFixed(3)} '
          'cx=${(detection.cx / imageWidth).toStringAsFixed(3)} '
          'cy=${(detection.cy / imageHeight).toStringAsFixed(3)} '
          'width=${(detection.width / imageWidth).toStringAsFixed(3)} '
          'height=${(detection.height / imageHeight).toStringAsFixed(3)}',
        );
      }
    }
  }

  double _norm(num value) => value / inputSize;
}

class _ModelBox {
  const _ModelBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  final int x;
  final int y;
  final int width;
  final int height;

  double get cx => x + width / 2.0;
  double get cy => y + height / 2.0;
  int get area => math.max(0, width) * math.max(0, height);
}

class _RecoveredClass {
  const _RecoveredClass(this.classId, this.confidence);

  final int classId;
  final double confidence;
}

_PreprocessedImage _preprocessImageForYolo(Uint8List imageBytes) {
  final original = img.decodeImage(imageBytes);
  if (original == null) {
    throw StateError('No se pudo decodificar la imagen local.');
  }

  final oriented = img.bakeOrientation(original);
  final resized = img.copyResize(
    oriented,
    width: BloodPressureDigitReader.inputSize,
    height: BloodPressureDigitReader.inputSize,
    interpolation: img.Interpolation.linear,
  );

  final input = Float32List(
    BloodPressureDigitReader.inputSize * BloodPressureDigitReader.inputSize * 3,
  );
  var index = 0;

  for (var y = 0; y < BloodPressureDigitReader.inputSize; y++) {
    for (var x = 0; x < BloodPressureDigitReader.inputSize; x++) {
      final pixel = resized.getPixel(x, y);
      input[index++] = pixel.r.toDouble() / 255.0;
      input[index++] = pixel.g.toDouble() / 255.0;
      input[index++] = pixel.b.toDouble() / 255.0;
    }
  }

  return _PreprocessedImage(
    input: input,
    imageWidth: oriented.width,
    imageHeight: oriented.height,
  );
}

class _PreprocessedImage {
  const _PreprocessedImage({
    required this.input,
    required this.imageWidth,
    required this.imageHeight,
  });

  final Float32List input;
  final int imageWidth;
  final int imageHeight;
}
