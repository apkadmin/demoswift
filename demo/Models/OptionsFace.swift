//
//  OptionsFace.swift
//  OCR-SDK
//
//  Created by annguyen on 12/03/2021.
//  Copyright © 2021 itsol. All rights reserved.
//

import Foundation
class OptionsFace {
    init(numClasses: Int, numBoxes: Int, numCoords: Int, keypointCoordOffset: Int, ignoreClasses: [Int], scoreClippingThresh: Double, minScoreThresh: Double, numKeypoints: Int, numValuesPerKeypoint: Int, boxCoordOffset: Int, xScale: Double, yScale: Double, wScale: Double, hScale: Double, applyExponentialOnBoxSize: Bool, reverseOutputOrder: Bool, sigmoidScore: Bool, flipVertically: Bool) {
        self.numClasses = numClasses
        self.numBoxes = numBoxes
        self.numCoords = numCoords
        self.keypointCoordOffset = keypointCoordOffset
        self.ignoreClasses = ignoreClasses
        self.scoreClippingThresh = scoreClippingThresh
        self.minScoreThresh = minScoreThresh
        self.numKeypoints = numKeypoints
        self.numValuesPerKeypoint = numValuesPerKeypoint
        self.boxCoordOffset = boxCoordOffset
        self.xScale = xScale
        self.yScale = yScale
        self.wScale = wScale
        self.hScale = hScale
        self.applyExponentialOnBoxSize = applyExponentialOnBoxSize
        self.reverseOutputOrder = reverseOutputOrder
        self.sigmoidScore = sigmoidScore
        self.flipVertically = flipVertically
    }
    
  var numClasses: Int
    var numBoxes: Int
    var numCoords: Int
    var keypointCoordOffset : Int
    var ignoreClasses: [Int]
    var scoreClippingThresh: Double
    var minScoreThresh: Double
    var numKeypoints: Int
    var numValuesPerKeypoint: Int
    var boxCoordOffset: Int
    var xScale: Double
    var yScale: Double
    var wScale : Double
    var hScale: Double
    var applyExponentialOnBoxSize: Bool
    var reverseOutputOrder: Bool
    var sigmoidScore: Bool
    var flipVertically: Bool

}

class Anchor {
    init(xCenter: Double, yCenter: Double, h: Double, w: Double) {
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.h = h
        self.w = w
    }
    var xCenter: Double
    var yCenter: Double
    var h: Double
    var w: Double
}
