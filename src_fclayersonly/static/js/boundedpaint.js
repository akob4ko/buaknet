/**
 * 1. Image is centered by computing center of a mass.
 * 2. Then it is scale down into 20x20 frame
 * 3. and centered relative to 28x28 window.
 */


class Point {
  constructor(x, y) {
    this._x = x;
    this._y = y;
  }

  get x() {
    return this._x;
  }

  get y() {
    return this._y;
  }

  set x(x) {
    this._x = x;
  }

  set y(y) {
    this._y = y;
  }

  set(x, y) {
    this._x = x;
    this._y = y;
  }
}

// make a class for the mouse data
class Mouse extends Point {
  constructor() {
    super(0, 0);
    this._down = false;
    this._px = 0;
    this._py = 0;
  }

  get down() {
    return this._down;
  }

  set down(d) {
    this._down = d;
  }

  get x() {
    return this._x;
  }

  get y() {
    return this._y;
  }

  set x(x) {
    this._x = x;
  }

  set y(y) {
    this._y = y;
  }

  get px() {
    return this._px;
  }

  get py() {
    return this._py;
  }

  set px(px) {
    this._px = px;
  }

  set py(py) {
    this._py = py;
  }

}

class Atrament {
  constructor(selector, width, height, color) {


    if (!document) throw new Error('no DOM found');

    // get canvas element
    if (selector instanceof window.Node && selector.tagName === 'CANVAS') this.canvas = selector;
    else if (typeof selector === 'string') this.canvas = document.querySelector(selector);
    else throw new Error(`can't look for canvas based on '${selector}'`);
    if (!this.canvas) throw new Error('canvas not found');

    // set external canvas params
    this.canvas.width = width || 250;
    this.canvas.height = height || 250;


    // create a mouse object
    this.mouse = new Mouse();

    // mousemove handler
    const mouseMove = (e) => {
      e.preventDefault();

       // Get position relative to the viewport.
      const rect = this.canvas.getBoundingClientRect();

      // Get position relative to the
      const position = e.changedTouches && e.changedTouches[0] || e;
      let x = position.offsetX;
      let y = position.offsetY;

      if (typeof x === 'undefined') {
        x = position.clientX // Horizontal coordinate of the mouse pointer
          + document.documentElement.scrollLeft
          - rect.left; // Distance between left edge of the html element and left edghe of the canvas.
      }
      if (typeof y === 'undefined') {
        y = position.clientY + document.documentElement.scrollTop - rect.top;
      }

      // draw if we should draw
      if (this.mouse.down) {
        this.draw(x, y);
        if (!this._dirty && (x !== this.mouse.x || y !== this.mouse.y)) {
          this._dirty = true;
          this.fireDirty();
        }
      }
      else {
        this.mouse.x = x;
        this.mouse.y = y;
      }
    };

    // mousedown handler
    const mouseDown = (mousePosition) => {
      mousePosition.preventDefault();
      // update position just in case
      mouseMove(mousePosition);

      // if we are filling - fill and return
      if (this._mode === 'fill') {
        this.fill();
        return;
      }

      // remember it
      this.mouse.px = this.mouse.x;
      this.mouse.py = this.mouse.y;
      // begin drawing
      this.mouse.down = true;
      this.context.beginPath();
      this.context.moveTo(this.mouse.px, this.mouse.py);
    };
    const mouseUp = () => {

      this.mouse.down = false;

      // stop drawing
      this.context.closePath();

      // Scale donw.

    };

    // Attach listeners which support both mouse and touch device.
    this.canvas.addEventListener('mousemove', mouseMove);
    this.canvas.addEventListener('mousedown', mouseDown);
    document.addEventListener('mouseup', mouseUp);
    this.canvas.addEventListener('touchstart', mouseDown);
    this.canvas.addEventListener('touchend', mouseUp);
    this.canvas.addEventListener('touchmove', mouseMove);

    // set internal canvas params
    this.context = this.canvas.getContext('2d');
    this.context.globalCompositeOperation = 'source-over';
    this.context.globalAlpha = 1;
    this.context.strokeStyle = color || 'rgba(0,0,0,1)';
    this.context.lineCap = 'round';
    this.context.lineJoin = 'round';
    this.context.translate(0.5, 0.5);

    this._filling = false;
    this._fillStack = [];

    // set drawing params
    this.SMOOTHING_INIT = 0.2;
    this.WEIGHT_SPREAD = 10;
    this._smoothing = this.SMOOTHING_INIT;
    this._maxWeight = 12;
    this._thickness = 2;
    this._targetThickness = 2;
    this._weight = 2;
    this._mode = 'draw';
    this._adaptive = true;
  }

  static lineDistance(x1, y1, x2, y2) {
    // calculate euclidean distance between (x1, y1) and (x2, y2)
    const xs = Math.pow(x2 - x1, 2);
    const ys = Math.pow(y2 - y1, 2);
    return Math.sqrt(xs + ys);
  }

  static hexToRgb(hexColor) {
    // Since input type color provides hex and ImageData accepts RGB need to transform
    const m = hexColor.match(/^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i);
    return [
      parseInt(m[1], 16),
      parseInt(m[2], 16),
      parseInt(m[3], 16)
    ];
  }

  static matchColor(data, compR, compG, compB, compA) {
    return (pixelPos) => {
      // Pixel color equals comp color?
      const r = data[pixelPos];
      const g = data[pixelPos + 1];
      const b = data[pixelPos + 2];
      const a = data[pixelPos + 3];

      return (r === compR && g === compG && b === compB && a === compA);
    };
  }

  static colorPixel(data, fillR, fillG, fillB, startColor, alpha) {
    const matchColor = Atrament.matchColor(data, ...startColor);

    return (pixelPos) => {
      // Update fill color in matrix
      data[pixelPos] = fillR;
      data[pixelPos + 1] = fillG;
      data[pixelPos + 2] = fillB;
      data[pixelPos + 3] = alpha;

      if (!matchColor(pixelPos + 4)) {
        data[pixelPos + 4] = data[pixelPos + 4] * 0.01 + fillR * 0.99;
        data[pixelPos + 4 + 1] = data[pixelPos + 4 + 1] * 0.01 + fillG * 0.99;
        data[pixelPos + 4 + 2] = data[pixelPos + 4 + 2] * 0.01 + fillB * 0.99;
        data[pixelPos + 4 + 3] = data[pixelPos + 4 + 3] * 0.01 + alpha * 0.99;
      }

      if (!matchColor(pixelPos - 4)) {
        data[pixelPos - 4] = data[pixelPos - 4] * 0.01 + fillR * 0.99;
        data[pixelPos - 4 + 1] = data[pixelPos - 4 + 1] * 0.01 + fillG * 0.99;
        data[pixelPos - 4 + 2] = data[pixelPos - 4 + 2] * 0.01 + fillB * 0.99;
        data[pixelPos - 4 + 3] = data[pixelPos - 4 + 3] * 0.01 + alpha * 0.99;
      }
    };
  }

  draw(mX, mY) {
    const mouse = this.mouse;
    const context = this.context;

    // calculate distance from previous point
    const rawDist = Atrament.lineDistance(mX, mY, mouse.px, mouse.py);

    // now, here we scale the initial smoothing factor by the raw distance
    // this means that when the mouse moves fast, there is more smoothing
    // and when we're drawing small detailed stuff, we have more control
    // also we hard clip at 1
    const smoothingFactor = Math.min(0.87, this._smoothing + (rawDist - 60) / 3000);

    // calculate smoothed coordinates
    mouse.x = mX - (mX - mouse.px) * smoothingFactor;
    mouse.y = mY - (mY - mouse.py) * smoothingFactor;

    // recalculate distance from previous point, this time relative to the smoothed coords
    const dist = Atrament.lineDistance(mouse.x, mouse.y, mouse.px, mouse.py);

    if (this._adaptive) {
      // calculate target thickness based on the new distance
      this._targetThickness = (dist - 1) / (50 - 1) * (this._maxWeight - this._weight) + this._weight;
      // approach the target gradually
      if (this._thickness > this._targetThickness) {
        this._thickness -= 0.5;
      }
      else if (this._thickness < this._targetThickness) {
        this._thickness += 0.5;
      }
      // set line width
      context.lineWidth = 15;
    }
    else {
      // line width is equal to default weight
      context.lineWidth = this._weight;
    }

    // draw using quad interpolation
    context.quadraticCurveTo(mouse.px, mouse.py, mouse.x, mouse.y);
    context.stroke();

    // remember
    mouse.px = mouse.x;
    mouse.py = mouse.y;
  }

  get weight() {
    return this._weight;
  }

  set weight(w) {
    if (typeof w !== 'number') throw new Error('wrong argument type');
    this._weight = w;
    this._thickness = w;
    this._targetThickness = w;
    this._maxWeight = w + this.WEIGHT_SPREAD;
  }

  get mode() {
    return this._mode;
  }

  get dirty() {
    return !!this._dirty;
  }

  set mode(m) {
    if (typeof m !== 'string') throw new Error('wrong argument type');
    switch (m) {
      case 'erase':
        this._mode = 'erase';
        this.context.globalCompositeOperation = 'destination-out';
        break;
      case 'fill':
        this._mode = 'fill';
        this.context.globalCompositeOperation = 'source-over';
        break;
      default:
        this._mode = 'draw';
        this.context.globalCompositeOperation = 'source-over';
        break;
    }
  }

  set opacity(o) {
    if (typeof o !== 'number') throw new Error('wrong argument type');
    // now, we need to scale this, because our drawing method means we don't just get uniform transparency all over the drawn line.
    // so we scale it down a lot, meaning that it'll look nicely semi-transparent
    // unless opacity is 1, then we should go full on to 1
    if (o >= 1) this.context.globalAlpha = 1;
    else this.context.globalAlpha = o / 10;
  }

  fireDirty() {
    const event = document.createEvent('Event');
    event.initEvent('dirty', true, true);
    this.canvas.dispatchEvent(event);
  }

  clear_letter() {
    var relatedField = document.getElementById('data-paint_letter');
      if (relatedField) {
        relatedField.value = "";
      }
      var divletters = $('div#res_letter');
      divletters.empty();
    if (!this.dirty) {
      return;
    }
    this._dirty = false;
    this.fireDirty();

    // make sure we're in the right compositing mode, and erase everything
    if (this.context.globalCompositeOperation === 'destination-out') {
      this.mode = 'draw';
      this.context.clearRect(-10, -10, this.canvas.width + 20, this.canvas.height + 20);
      this.mode = 'erase';
    }
    else {
      this.context.clearRect(-10, -10, this.canvas.width + 20, this.canvas.height + 20);
    }
  }
  clear_original_letter() {
    var relatedField = document.getElementById('data-paint_letter');
      if (relatedField) {
        relatedField.value = "";
      }
    if (!this.dirty) {
      return;
    }
    this._dirty = false;
    this.fireDirty();

    // make sure we're in the right compositing mode, and erase everything
    if (this.context.globalCompositeOperation === 'destination-out') {
      this.mode = 'draw';
      this.context.clearRect(-10, -10, this.canvas.width + 20, this.canvas.height + 20);
      this.mode = 'erase';
    }
    else {
      this.context.clearRect(-10, -10, this.canvas.width + 20, this.canvas.height + 20);
    }
  }
  clear() {
    var relatedField = document.getElementById('data-paint');
      if (relatedField) {
        relatedField.value = "";
      }
      var spanr = $('span#result');
      var spanp = $('span#probability');
      spanr.text('');
      spanp.text('');
    if (!this.dirty) {
      return;
    }
    this._dirty = false;
    this.fireDirty();

    // make sure we're in the right compositing mode, and erase everything
    if (this.context.globalCompositeOperation === 'destination-out') {
      this.mode = 'draw';
      this.context.clearRect(-10, -10, this.canvas.width + 20, this.canvas.height + 20);
      this.mode = 'erase';
    }
    else {
      this.context.clearRect(-10, -10, this.canvas.width + 20, this.canvas.height + 20);
    }
    this.scaleDown();
  }

  fill() {
    const mouse = this.mouse;
    const context = this.context;
    const startColor = Array.prototype.slice.call(context.getImageData(mouse.x, mouse.y, 1, 1).data, 0); // converting to Array because Safari 9

    if (!this._filling) {
      this.canvas.style.cursor = 'progress';
      this._filling = true;
      setTimeout(() => { this._floodFill(mouse.x, mouse.y, startColor); }, 100);
    }
    else {
      this._fillStack.push([
        mouse.x,
        mouse.y,
        startColor
      ]);
    }
  }

  _floodFill(startX, startY, startColor) {
    const context = this.context;
    const canvasWidth = context.canvas.width;
    const canvasHeight = context.canvas.height;
    const pixelStack = [[startX, startY]];
    // hex needs to be trasformed to rgb since colorLayer accepts RGB
    const fillColor = Atrament.hexToRgb(this.color);

    // Need to save current context with colors, we will update it
    const colorLayer = context.getImageData(0, 0, context.canvas.width, context.canvas.height);


    const alpha = Math.min(context.globalAlpha * 10 * 255, 255);
    const colorPixel = Atrament.colorPixel(colorLayer.data, ...fillColor, startColor, alpha);
    const matchColor = Atrament.matchColor(colorLayer.data, ...startColor);
    const matchFillColor = Atrament.matchColor(colorLayer.data, ...[...fillColor, 255]);

    // check if we're trying to fill with the same colour, if so, stop
    if (matchFillColor((startY * context.canvas.width + startX) * 4)) {
      this._filling = false;
      setTimeout(() => { this.canvas.style.cursor = 'crosshair'; }, 100);
      return;
    }

    while (pixelStack.length) {
      const newPos = pixelStack.pop();
      const x = newPos[0];
      let y = newPos[1];

      let pixelPos = (y * canvasWidth + x) * 4;

      while (y-- >= 0 && matchColor(pixelPos)) {
        pixelPos -= canvasWidth * 4;
      }
      pixelPos += canvasWidth * 4;

      ++y;

      let reachLeft = false;
      let reachRight = false;

      while (y++ < canvasHeight - 1 && matchColor(pixelPos)) {
        colorPixel(pixelPos);

        if (x > 0) {
          if (matchColor(pixelPos - 4)) {
            if (!reachLeft) {
              pixelStack.push([x - 1, y]);
              reachLeft = true;
            }
          }
          else if (reachLeft) {
            reachLeft = false;
          }
        }

        if (x < canvasWidth - 1) {
          if (matchColor(pixelPos + 4)) {
            if (!reachRight) {
              pixelStack.push([x + 1, y]);
              reachRight = true;
            }
          }
          else if (reachRight) {
            reachRight = false;
          }
        }

        pixelPos += canvasWidth * 4;
      }
    }

    // Update context with filled bucket!
    context.putImageData(colorLayer, 0, 0);

    if (this._fillStack.length) {
      this._floodFill(...this._fillStack.shift());
    }
    else {
      this._filling = false;
      setTimeout(() => { this.canvas.style.cursor = 'crosshair'; }, 100);
    }
  }


  scaleDown() {
    const originalCanvas = document.getElementById('original');
    const boundedCanvas = document.getElementById('bounded');
    const mnistCanvas = document.getElementById('mnist');

    const originalCtx = originalCanvas.getContext('2d');
    const boundedCtx = boundedCanvas.getContext('2d');
    const mnistCtx = mnistCanvas.getContext('2d');


    const imgData = originalCtx.getImageData(0, 0, originalCanvas.width, originalCanvas.height);

    const data = imgData.data;
    const pixels = [[]];

    let row = 0;
    let column = 0;
    for (let i = 0; i < originalCanvas.width * originalCanvas.height * 4; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3] / 255;

      if (column >= originalCanvas.width) {
        column = 0;
        row++;
        pixels[row] = [];
      }

      pixels[row].push(Math.round(a * 250) / 250);

      column++;

    }

    const boundingRectangle = this.getBoundingRectangle(pixels);


    const array = pixels.reduce(
      (arr, row) => [].concat(arr, row.reduce(
        (concatedRow, alpha) => [].concat(concatedRow, [0, 0, 0, alpha * 255]), []))
    , []);


    const clampedArray =  new Uint8ClampedArray(array);
    const bounded = new ImageData(clampedArray, 250, 250);

    boundedCtx.putImageData(bounded, 0, 0);

    boundedCtx.beginPath();
    boundedCtx.lineWidth= '1';
    boundedCtx.strokeStyle= 'red';
    boundedCtx.rect(
      boundingRectangle.minX,
      boundingRectangle.minY,
      Math.abs(boundingRectangle.minX - boundingRectangle.maxX),
      Math.abs(boundingRectangle.minY - boundingRectangle.maxY),
    );
    boundedCtx.stroke();



    var brW = boundingRectangle.maxX + 1 - boundingRectangle.minX;
    var brH = boundingRectangle.maxY + 1 - boundingRectangle.minY;
    const scalingFactor = 20 / Math.max(brW, brH);




    // Reset the drawing.
    var img = mnistCtx.createImageData(100, 100);
    for (var i = img.data.length; --i >= 0; )
      img.data[i] = 0;
    mnistCtx.putImageData(img, 100, 100);

    // Reset the tranforms
    mnistCtx.setTransform(1, 0, 0, 1, 0, 0);

    // Clear the canvas.
    mnistCtx.clearRect(0, 0, mnistCanvas.width, mnistCanvas.height);

    mnistCtx.beginPath();

    // mnistCtx.scale(scalingFactor, scalingFactor);
    mnistCtx.translate(
      -brW * scalingFactor / 2,
      -brH * scalingFactor / 2
    );
    mnistCtx.translate(
      mnistCtx.canvas.width / 2,
      mnistCtx.canvas.height / 2
    );
    mnistCtx.translate(
      -Math.min(boundingRectangle.minX, boundingRectangle.maxX) * scalingFactor,
      -Math.min(boundingRectangle.minY, boundingRectangle.maxY) * scalingFactor
    );
    mnistCtx.scale(scalingFactor, scalingFactor);

    mnistCtx.drawImage(originalCtx.canvas, 0, 0);
    mnistCtx.fillRelatedField = function () {
      document.getElementById('data-paint').value = mnistCanvas.toDataURL();
    };
    mnistCtx.fillRelatedField()
  }
  scaleDown_letter() {
    const originalCanvas = document.getElementById('original_letter');
    const boundedCanvas = document.getElementById('bounded_letter');
    const mnistCanvas = document.getElementById('mnist_letter');

    const originalCtx = originalCanvas.getContext('2d');
    const boundedCtx = boundedCanvas.getContext('2d');
    const mnistCtx = mnistCanvas.getContext('2d');


    const imgData = originalCtx.getImageData(0, 0, originalCanvas.width, originalCanvas.height);

    const data = imgData.data;
    const pixels = [[]];

    let row = 0;
    let column = 0;
    for (let i = 0; i < originalCanvas.width * originalCanvas.height * 4; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3] / 255;

      if (column >= originalCanvas.width) {
        column = 0;
        row++;
        pixels[row] = [];
      }

      pixels[row].push(Math.round(a * 250) / 250);

      column++;

    }

    const boundingRectangle = this.getBoundingRectangle(pixels);


    const array = pixels.reduce(
      (arr, row) => [].concat(arr, row.reduce(
        (concatedRow, alpha) => [].concat(concatedRow, [0, 0, 0, alpha * 255]), []))
    , []);


    const clampedArray =  new Uint8ClampedArray(array);
    const bounded = new ImageData(clampedArray, 250, 250);

    boundedCtx.putImageData(bounded, 0, 0);

    boundedCtx.beginPath();
    boundedCtx.lineWidth= '1';
    boundedCtx.strokeStyle= 'red';
    boundedCtx.rect(
      boundingRectangle.minX,
      boundingRectangle.minY,
      Math.abs(boundingRectangle.minX - boundingRectangle.maxX),
      Math.abs(boundingRectangle.minY - boundingRectangle.maxY),
    );
    boundedCtx.stroke();



    var brW = boundingRectangle.maxX + 1 - boundingRectangle.minX;
    var brH = boundingRectangle.maxY + 1 - boundingRectangle.minY;
    const scalingFactor = 20 / Math.max(brW, brH);




    // Reset the drawing.
    var img = mnistCtx.createImageData(100, 100);
    for (var i = img.data.length; --i >= 0; )
      img.data[i] = 0;
    mnistCtx.putImageData(img, 100, 100);

    // Reset the tranforms
    mnistCtx.setTransform(1, 0, 0, 1, 0, 0);

    // Clear the canvas.
    mnistCtx.clearRect(0, 0, mnistCanvas.width, mnistCanvas.height);

    mnistCtx.beginPath();

    // mnistCtx.scale(scalingFactor, scalingFactor);
    mnistCtx.translate(
      -brW * scalingFactor / 2,
      -brH * scalingFactor / 2
    );
    mnistCtx.translate(
      mnistCtx.canvas.width / 2,
      mnistCtx.canvas.height / 2
    );
    mnistCtx.translate(
      -Math.min(boundingRectangle.minX, boundingRectangle.maxX) * scalingFactor,
      -Math.min(boundingRectangle.minY, boundingRectangle.maxY) * scalingFactor
    );
    mnistCtx.scale(scalingFactor, scalingFactor);

    mnistCtx.drawImage(originalCtx.canvas, 0, 0);
    mnistCtx.fillRelatedField()
  }

  // Function takes grayscale image and finds bounding rectangle of digit defined by corresponding treshold.
  getBoundingRectangle(img, threshold = 0.01) {
    const rows = img.length;
    const columns= img[0].length;

    let minX = columns;
    let minY = rows;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < columns; x++) {
        if (img[y][x] > 1 -  threshold) {
          if (minX > x) minX = x;
          if (maxX < x) maxX = x;
          if (minY > y) minY = y;
          if (maxY < y) maxY = y;
        }
      }
    }
    return { minY, minX, maxY, maxX };
  }

  /**
   * Evaluates center of mass of digit, in order to center it.
   * Note that 1 stands for black and 0 for white so it has to be inverted.
   */
  centerImage(img) {
    var
      meanX     = 0,
      meanY     = 0,
      rows      = img.length,
      columns   = img[0].length,
      pixel     = 0,
      sumPixels = 0,
      y         = 0,
      x         = 0;

    for (y = 0; y < rows; y++) {
      for (x = 0; x < columns; x++) {
        // pixel = (1 - img[y][x]);
        pixel = img[y][x];
        sumPixels += pixel;
        meanY += y * pixel;
        meanX += x * pixel;
      }
    }
    meanX /= sumPixels;
    meanY /= sumPixels;

    const dY = Math.round(rows/2 - meanY);
    const dX = Math.round(columns/2 - meanX);

    return {transX: dX, transY: dY};
  }
}

var sketcher = new Atrament('#original',250,250)
$('#post_data').on('click', (e) => {
  const mnistCanvas = document.getElementById('mnist');
  mnistCanvas.style.border ="1px solid #000000";
  sketcher.scaleDown();
});
$('#clear').on('click', (e) => {
  const mnistCanvas = document.getElementById('mnist');
  mnistCanvas.style.border ="0px solid #000000";
  sketcher.clear();
});

$('a#post_data').bind('click', function() {
            $.ajax({
                url : $root + '/get_digit',
                type: "POST",
                data: JSON.stringify([
                    {digit: $('textarea[id="data-paint"]').val()}
                ]),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
                success: function(data){
                    $("#result").text(data.result);
                    $("#probability").text(data.probability);
                }
            });
            return false;
});
