//
//  FaceDetection.swift
//  OCR-SDK
//
//  Created by annguyen on 12/03/2021.
//  Copyright © 2021 itsol. All rights reserved.
//

import Foundation
import UIKit
class FaceDetection{
    public func getAnchors(options: AnchorOption) -> [Anchor] {
        var _anchors: [Anchor]  = []
        if (options.stridesSize() != options.numLayers) {
            print("strides_size and num_layers must be equal.")
            return []
        }
        var layerID: Int = 0
        while (layerID < options.stridesSize()) {
            var anchorHeight: [Double] = []
            var anchorWidth: [Double] = []
            var aspectRatios: [Double] = []
            var scales: [Double] = []
            var lastSameStrideLayer: Int = layerID
            while (lastSameStrideLayer < options.stridesSize() &&
                    options.strides[lastSameStrideLayer] == options.strides[layerID]) {
                let scale: Double = options.minScale + (options.maxScale - options.minScale) * Double(lastSameStrideLayer) / (Double(options.stridesSize()) - 1.0)
                
                if (lastSameStrideLayer == 0 && options.reduceBoxesInLowestLayer) {
                    aspectRatios.append(1.0)
                    aspectRatios.append(2.0)
                    aspectRatios.append(0.5)
                    scales.append(0.1)
                    scales.append(scale)
                    scales.append(scale)
                } else {
                    for i in 0..<options.aspectRatios.count {
                        aspectRatios.append(options.aspectRatios[i])
                        scales.append(scale)
                    }
                    
                    if options.interpolatedScaleAspectRatio > 0 {
                        var scaleNext: Double = 0.0
                        if lastSameStrideLayer == (options.stridesSize() - 1) {
                            scaleNext = 1.0
                        } else {
                            scaleNext = options.minScale + (options.maxScale - options.minScale) + Double(lastSameStrideLayer + 1) / Double(options.stridesSize() - 1)
                        }
                        scales.append(sqrt(scale * scaleNext))
                        aspectRatios.append(options.interpolatedScaleAspectRatio)
                    }
                }
                lastSameStrideLayer += 1
            }
            for i in 0..<aspectRatios.count {
                let ratioSQRT: Double = sqrt(aspectRatios[i])
                anchorHeight.append(scales[i] / ratioSQRT)
                anchorWidth.append(scales[i] * ratioSQRT)
            }
            var featureMapHeight: Int = 0
            var featureMapWidth: Int = 0
            if (options.featureMapHeightSize() > 0) {
                featureMapHeight = options.featureMapHeight[layerID]
                featureMapWidth = options.featureMapWidth[layerID]
            } else {
                let stride: Int = options.strides[layerID]
                featureMapHeight = Int(options.inputSizeHeight / stride)
                featureMapWidth = Int(options.inputSizeWidth / stride)
            }
            
            for y in 0..<featureMapHeight {
                for x: Int in 0..<featureMapWidth {
                    for anchorID in 0..<anchorHeight.count {
                        let xCenter: Double = Double(Double(x) + options.anchorOffsetX) / Double(featureMapWidth)
                        let yCenter: Double = Double(Double(y) + options.anchorOffsetY) / Double(featureMapHeight)
                        var w: Double = 0
                        var h: Double = 0
                        if (options.fixedAnchorSize) {
                            w = 1.0
                            h = 1.0
                        } else {
                            w = anchorWidth[anchorID]
                            h = anchorHeight[anchorID]
                        }
                        _anchors.append(Anchor(xCenter: xCenter, yCenter: yCenter, h: h, w: w))
                    }
                }
            }
            layerID = lastSameStrideLayer
        }
        return _anchors
    }
    
    //1 classificators: 2: regression
    func process(options: OptionsFace, rawScores: [Double] , rawBoxes: [Double] , anchors: [Anchor]) -> [Detection] {
        var detectionScores: [Double] = []
        var detectionClasses: [Int] = []
        let boxes = options.numBoxes
        for i in 0..<boxes {
            var classId = -1
            var maxScore: Double =  5e-324
            for scoreIdx in 0..<options.numClasses {
                var score = rawScores[i * options.numClasses + scoreIdx]
                if options.sigmoidScore {
                    if options.scoreClippingThresh > 0 {
                        if score < -options.scoreClippingThresh {
                            score = -options.scoreClippingThresh
                        }
                        if score > options.scoreClippingThresh {
                            score = options.scoreClippingThresh
                          
                        }
                        score = 1.0 / (1.0 + exp(-score))
                        if maxScore < score {
                            maxScore = score
                            classId = scoreIdx
                        }
                    }
                }
            }
            detectionClasses.append(classId)
            detectionScores.append(maxScore)
        }
        let  detections: [Detection] = convertToDetections(rawBoxes: rawBoxes, anchors: anchors, detectionScores: detectionScores, detectionClasses: detectionClasses, options: options)
        return detections
    }
    
    func convertToDetections(
        rawBoxes: [Double],
        anchors: [Anchor],
        detectionScores: [Double],
        detectionClasses: [Int],
        options: OptionsFace) ->  [Detection]{
        var _outputDetections : [Detection] = []
        for i in 0..<options.numBoxes {
            if detectionScores[i] < options.minScoreThresh {
                continue
            }

            let boxOffset: Int = 0
            let boxData = decodeBox(rawBoxes: rawBoxes, i:i, anchors: anchors, options: options)
            var landmark: [Landmark] = []
            for k in 0..<options.numKeypoints {
                let x: Double = boxData[boxOffset + 4 + k * 2]
                var y: Double = 0.0
                if (options.flipVertically) {
                    y = 1 - boxData[boxOffset + 4 + k * 2 + 1]
                } else {
                    y = boxData[boxOffset + 4 + k * 2 + 1]
                }
                let tmpLand: Landmark = Landmark(x: x, y: y)
                landmark.append(tmpLand)
            }
            let detection: Detection = convertToDetection(
                boxYMin: boxData[boxOffset + 0],
                boxXMin: boxData[boxOffset + 1],
                boxYMax: boxData[boxOffset + 2],
                boxXMax: boxData[boxOffset + 3],
                landmark: landmark,
                score: detectionScores[i],
                classID: detectionClasses[i],
                flipVertically: options.flipVertically)
            
            _outputDetections.append(detection)
        }
        return _outputDetections
    }
    
    
    func origNms(detections: [Detection],threshold: Double, img_width: Int, img_height: Int) -> [Detection] {
        if detections.count <= 0 {
            return []
        }
        var x1: [Double] = []
        var x2: [Double] = []
        var y1: [Double] = []
        var y2: [Double] = []
        var s : [Double] = []
        
        detections.forEach{detection in
            x1.append(detection.xMin * Double(img_width))
            x2.append((detection.xMin + detection.width) * Double(img_width))
            y1.append(detection.yMin * Double(img_height))
            y2.append((detection.yMin + detection.height) * Double(img_height))
            s.append(detection.score)
        }
        
        let _x1 = x1
        let _x2 = x2
        let _y1 = y1
        let _y2 = y2
        
        let area: [Double] =  multiplyArray(x: subArray(x: _x2, y: _x1) , y: subArray(x: _y2 , y: _y1))
        
        let I: [Double] = _quickSort(a: s)
        
        var positions: [Int] = []
        I.forEach{element in
            positions.append(s.firstIndex(of: element)!)
        }
        
        var pick: [Int] = []
        while positions.count > 0 {
            let ind0: [Int] = [positions.last!]
            let ind1: [Int] = positions.dropLast()
            
            let xx1 = _maximum(value: _itemIndex(item: _x1, positions: ind0)[0], itemIndex: _itemIndex(item: _x1, positions: ind1))
            let yy1 = _maximum(value: _itemIndex(item: _y1, positions: ind0)[0], itemIndex: _itemIndex(item: _y1, positions: ind1))
            let xx2 = _minimum(value: _itemIndex(item: _x2, positions: ind0)[0], itemIndex: _itemIndex(item: _x2, positions: ind1))
            let yy2 = _minimum(value: _itemIndex(item: _y2, positions: ind0)[0], itemIndex: _itemIndex(item: _y2, positions: ind1))
            
            let w = _maximum(value: 0.0, itemIndex: subArray(x: xx2 ,y: xx1))
            let h = _maximum(value: 0.0, itemIndex: subArray(x: yy2, y: yy1))
            
            let inter = multiplyArray(x: w, y: h)
            let o = divideArray(x: inter,
                                y: subArray(x: _sum(a: _itemIndex(item: area, positions: ind0)[0], b: _itemIndex(item: area, positions: ind1)), y: inter))
            
            pick.append(ind0[0])
            let _inCorrectIndex: [Int] = inCorrectIndex(positions: positions, o: o, threshold: threshold)
            positions = removeInCorrectIndex(positions: positions, inCorrectIndex: _inCorrectIndex)
        }
        var _detections: [Detection] = []
        pick.forEach{element in _detections.append(detections[element])}
        return _detections
    }
    
}

func subArray(x: [Double], y:[Double]) -> [Double] {
    var a: [Double] = []
    for b in 0..<x.count {
        a.append(x[b] - y[b])
    }
    return a
}
func multiplyArray(x: [Double], y:[Double]) -> [Double] {
    var a: [Double] = []
    for b in 0..<x.count {
        a.append(x[b] * y[b])
    }
    return a
}
func divideArray(x: [Double], y: [Double]) -> [Double] {
    var a: [Double] = []
    for b in 0..<x.count {
        a.append(x[b] / y[b])
    }
    return a
}

func decodeBox(rawBoxes: [Double], i: Int, anchors: [Anchor], options: OptionsFace) -> [Double] {
    var boxData: [Double] =  [Double](repeating: 0.0, count: options.numCoords)
    let boxOffset: Int = i * options.numCoords + options.boxCoordOffset
    var yCenter: Double = rawBoxes[boxOffset]
    var xCenter: Double = rawBoxes[boxOffset + 1]
    var h: Double = rawBoxes[boxOffset + 2]
    var w: Double = rawBoxes[boxOffset + 3]
    if (options.reverseOutputOrder) {
        xCenter = rawBoxes[boxOffset]
        yCenter = rawBoxes[boxOffset + 1]
        w = rawBoxes[boxOffset + 2]
        h = rawBoxes[boxOffset + 3]
    }
    
    xCenter = xCenter / options.xScale * anchors[i].w + anchors[i].xCenter
    yCenter = yCenter / options.yScale * anchors[i].h + anchors[i].yCenter
    
    if (options.applyExponentialOnBoxSize) {
        h = exp(h / options.hScale) * anchors[i].h
        w = exp(w / options.wScale) * anchors[i].w
    } else {
        h = h / options.hScale * anchors[i].h
        w = w / options.wScale * anchors[i].w
    }
    
    let yMin: Double = yCenter - h / 2.0
    let xMin: Double = xCenter - w / 2.0
    let yMax: Double = yCenter + h / 2.0
    let xMax: Double = xCenter + w / 2.0
    
    boxData[0] = yMin
    boxData[1] = xMin
    boxData[2] = yMax
    boxData[3] = xMax
    
    if (options.numKeypoints > 0) {
        for k in 0..<options.numKeypoints {
            let offset: Int = i * options.numCoords +
                options.keypointCoordOffset +
                k * options.numValuesPerKeypoint
            var keyPointY: Double = rawBoxes[offset]
            var keyPointX: Double = rawBoxes[offset + 1]
            
            if (options.reverseOutputOrder) {
                keyPointX = rawBoxes[offset]
                keyPointY = rawBoxes[offset + 1]
            }
            boxData[4 + k * options.numValuesPerKeypoint] =
                keyPointX / options.xScale * anchors[i].w + anchors[i].xCenter
            
            boxData[4 + k * options.numValuesPerKeypoint + 1] =
                keyPointY / options.yScale * anchors[i].h + anchors[i].yCenter
        }
    }
    return boxData
}


func convertToDetection(
    boxYMin: Double,
    boxXMin: Double,
    boxYMax: Double,
    boxXMax: Double,
    landmark: [Landmark],
    score: Double,
    classID: Int,
    flipVertically: Bool) -> Detection {
    var _yMin: Double = 0.0
    if flipVertically {
        _yMin = 1.0 - boxYMax
    }
    else {
        _yMin = boxYMin
    }
    
    return Detection(score: score, xMin: boxXMin, yMin: _yMin, width: (boxXMax - boxXMin), height: (boxYMax - boxYMin), classID: classID, landMark: landmark)
}

func clamp(lower: Int,higher: Int, val: Int) -> Int {
    if val < lower {
        return 0
    }
    else if val > higher {
        return 255
    }
    else {
        return val
    }
}

func getRotatedImageByteIndex(x: Int, y: Int, rotatedImageWidth: Int) -> Int {
    return rotatedImageWidth * (y + 1) - (x + 1)
}

func _quickSort(a: [Double]) -> [Double] {
    if a.count <= 1{
        return a
    }
    
    let pivot = a[0]
    var less: [Double] = []
    var more: [Double] = []
    var pivotList: [Double] = []
    
    a.forEach{i in
        if (i < pivot) {
            less.append(i)
        } else if (i > pivot) {
            more.append(i)
        } else {
            pivotList.append(i)
        }
    }
    
    
    less = _quickSort(a: less)
    more = _quickSort(a: more)
    
    less += pivotList
    less += more
    return less
}
func _itemIndex(item: [Double], positions:[Int]) -> [Double] {
    var _temp: [Double] = []
    positions.forEach {element in  _temp.append(item[element])}
    return _temp
}
func _minimum(value: Double, itemIndex: [Double]) -> [Double] {
    var _temp: [Double] = []
    itemIndex.forEach{element in
        if value < element {
            _temp.append(value)
        }
        else {
            _temp.append(element)
        }
    }
    return  _temp
}

func _maximum(value: Double, itemIndex: [Double]) -> [Double] {
    var _temp: [Double] = []
    itemIndex.forEach{element in
        if value > element {
            _temp.append(value)
        }
        else {
            _temp.append(element)
        }
    }
    return  _temp
}

func _sum(a: Double, b: [Double]) -> [Double] {
    var _temp: [Double] = []
    b.forEach{element in
        _temp.append(a + element)
    }
    return _temp
}

func inCorrectIndex(positions: [Int], o: [Double], threshold: Double) -> [Int] {
    var _index: [Int] = []
    for i in 0..<o.count {
        if o[i] > threshold {
            _index.append(positions[i])
        }
    }
    return _index
}
func removeInCorrectIndex(positions: [Int], inCorrectIndex: [Int]) -> [Int] {
    var temp = positions
    temp.remove(at: positions.count - 1)
    inCorrectIndex.forEach{ element in temp = temp.filter(){$0 != element}}
    return temp
}

//Uint32List convertImage(Uint8List plane0, Uint8List plane1, Uint8List plane2,
//    int bytesPerRow, int bytesPerPixel, int width, int height) {
//  int hexFF = 255
//  int x, y, uvIndex, index
//  int yp, up, vp
//  int r, g, b
//  int rt, gt, bt
//
//  Uint32List image = new Uint32List(width * height)
//
//  for (x = 0 x < width x++) {
//    for (y = 0 y < height y++) {
//      uvIndex =
//          bytesPerPixel * ((x / 2).round() + bytesPerRow * ((y / 2).round()))
//      index = y * width + x
//
//      yp = plane0[index]
//      up = plane1[uvIndex]
//      vp = plane2[uvIndex]
//
//      rt = (yp + vp * 1436 / 1024 - 179).round()
//      gt = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91).round()
//      bt = (yp + up * 1814 / 1024 - 227).round()
//      r = clamp(0, 255, rt)
//      g = clamp(0, 255, gt)
//      b = clamp(0, 255, bt)
//
//      image[getRotatedImageByteIndex(y, x, height)] =
//          (hexFF << 24) | (b << 16) | (g << 8) | r
//    }
//  }
//  return image
//}

//func FaceAlign(
//    rawImage: CVPixelBuffer, detections: [Detection],  width: Int) -> [Any] {
//    var desiredLeftEye: Landmark = Landmark(x: 0.35, y: 0.35)
//    var desiredFaceWidth: Int = width
//    var desiredFaceHeight: Int = width
//
//  imglib.PngEncoder pngEncoder = new imglib.PngEncoder(level: 0, filter: 0)
//  List<int> byteData = pngEncoder.encodeImage(rawImage)
//
//  Detection detection
//  List<dynamic> newFaces = new List()
//
//  for (detection in detections) {
//    Landmark leftEyeCenter = detection.landmark[0]
//    Landmark rightEyeCenter = detection.landmark[1]
//
//    double dY = (rightEyeCenter.y - leftEyeCenter.y) * rawImage.height
//    double dX = (rightEyeCenter.x - leftEyeCenter.x) * rawImage.width
//
//    double angle = atan2(dY, dX)
//    angle = (angle > 0 ? angle : (2 * pi + angle)) * 360 / (2 * pi)
//
//    double desiredRightEyeX = 1.0 - desiredLeftEye.x
//    double dist = sqrt((dX * dX) + (dY * dY))
//    double desiredDist = (desiredRightEyeX - desiredLeftEye.x)
//    desiredDist *= desiredFaceWidth
//
//    double scale = desiredDist / dist
//
//    double eyeCenterX =
//        ((leftEyeCenter.x + rightEyeCenter.x) / 2) * rawImage.width
//    double eyeCenterY =
//        ((leftEyeCenter.y + rightEyeCenter.y) / 2) * rawImage.height
//
//    List<int> eyeCenter = new List()
//    eyeCenter.add(eyeCenterX.round())
//    eyeCenter.add(eyeCenterY.round())
//
//    List<double> desiredLeftEye_push = new List()
//    desiredLeftEye_push.add(desiredLeftEye.x)
//    desiredLeftEye_push.add(desiredLeftEye.y)
//
//    List<int> dstSize = new List()
//    dstSize.add(desiredFaceWidth)
//    dstSize.add(desiredFaceHeight)
//    dynamic byteFace = await ImgProc.faceAlign(
//        byteData, eyeCenter, desiredLeftEye_push, angle, scale, dstSize)
//    newFaces.add(byteFace)
//  }
//  return newFaces
//}

