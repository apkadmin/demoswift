//
//  AnchorOption.swift
//  OCR-SDK
//
//  Created by annguyen on 12/03/2021.
//  Copyright © 2021 itsol. All rights reserved.
//

import Foundation
class AnchorOption {
    init(inputSizeWidth: Int, inputSizeHeight: Int, minScale: Double, maxScale: Double, anchorOffsetX: Double, anchorOffsetY: Double, numLayers: Int, featureMapWidth: [Int], featureMapHeight: [Int], strides: [Int], aspectRatios: [Double], reduceBoxesInLowestLayer: Bool, interpolatedScaleAspectRatio: Double, fixedAnchorSize: Bool) {
        self.inputSizeWidth = inputSizeWidth
        self.inputSizeHeight = inputSizeHeight
        self.minScale = minScale
        self.maxScale = maxScale
        self.anchorOffsetX = anchorOffsetX
        self.anchorOffsetY = anchorOffsetY
        self.numLayers = numLayers
        self.featureMapWidth = featureMapWidth
        self.featureMapHeight = featureMapHeight
        self.strides = strides
        self.aspectRatios = aspectRatios
        self.reduceBoxesInLowestLayer = reduceBoxesInLowestLayer
        self.interpolatedScaleAspectRatio = interpolatedScaleAspectRatio
        self.fixedAnchorSize = fixedAnchorSize
    }
    
    var inputSizeWidth: Int
    var inputSizeHeight: Int
    var minScale: Double
    var maxScale: Double
    var anchorOffsetX: Double
    var anchorOffsetY: Double
    var numLayers: Int
    var featureMapWidth: [Int]
    var featureMapHeight: [Int]
    var strides: [Int]
    var aspectRatios: [Double]
    var reduceBoxesInLowestLayer: Bool
    var interpolatedScaleAspectRatio: Double
    var fixedAnchorSize: Bool
    
    func stridesSize() -> Int {
        return strides.count
      }

    func featureMapHeightSize() -> Int {
        return featureMapHeight.count
      }

    func featureMapWidthSize() -> Int {
        return featureMapWidth.count
      }
}
