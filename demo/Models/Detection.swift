//
//  Detection.swift
//  OCR-SDK
//
//  Created by annguyen on 12/03/2021.
//  Copyright © 2021 itsol. All rights reserved.
//

import Foundation
struct Detection {
    init(score: Double, xMin: Double, yMin: Double, width: Double, height: Double, classID: Int, landMark: [Landmark]) {
        self.score = score
        self.xMin = xMin
        self.yMin = yMin
        self.width = width
        self.height = height
        self.classID = classID
        self.landMark = landMark
    }
    var score: Double
    var xMin: Double
    var yMin: Double
    var width: Double
    var height: Double
    var classID: Int
    var landMark: [Landmark]
}
