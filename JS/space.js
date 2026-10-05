class space{
	constructor(originX=0, originY=0, screenWidth=0, screenHeight=0, zoom=100){
		this.originX = originX; //left coordinate of the screen relative to x = 0
		this.originY = originY; //top coordinate of the screen relattive to y = 0
		this.screenWidth = screenWidth;
		this.screenHeight = screenHeight;
		this.zoom = zoom; // 100 = 100%, 10 = 10%
		this.scale = zoom / 100;
	}

	#drawGrid(ctx){
		ctx.save();
		ctx.clearRect(0, 0, this.screenWidth, this.screenHeight); // clear the grid
		ctx.beginPath(); // start drawing
		ctx.strokeStyle = "#141A24"; //white
                //ctx.strokeStyle = "#ccc"; // gray
		ctx.lineWidth = 1;

		for(let x = this.originX % this.zoom; x < this.screenWidth; x += this.zoom){ // draw vertical lines
			ctx.moveTo(x, 0); // from top
			ctx.lineTo(x, this.screenHeight); // to bottom
		}
		for(let y = this.originY % this.zoom; y < this.screenHeight; y += this.zoom){ // draw horizzontal lines
			ctx.moveTo(0, y); // from left
			ctx.lineTo(this.screenWidth, y); // to right
		}
		ctx.stroke();
		ctx.restore();
	}

	toScreen(absoluteX, absoluteY){
		const pixelsPerAU = 400;
		const relativeX = this.originX + (absoluteX * pixelsPerAU * this.scale);
    		const relativeY = this.originY + (absoluteY * pixelsPerAU * this.scale);

		return { x: relativeX, y: relativeY };
	}

	dragSpace(ctx, draggedX=0, draggedY=0){
		this.originX += draggedX;
		this.originY += draggedY;
		this.#drawGrid(ctx); // update grid
	}

	zoomSpace(ctx, zoomedAmount = 0) {
    const MIN_ZOOM = 1;
    const MAX_ZOOM = 150;
    const oldZoom = this.zoom;

    this.zoom *= 1 + zoomedAmount / 100;
    this.zoom = Math.max(MIN_ZOOM, Math.min(this.zoom, MAX_ZOOM));

    this.scale = this.zoom / 100;

    const scale = this.zoom / oldZoom;

    const centerX = this.screenWidth / 2;
    const centerY = this.screenHeight / 2;

    this.originX = centerX - (centerX - this.originX) * scale;
    this.originY = centerY - (centerY - this.originY) * scale;

    this.#drawGrid(ctx);
}
}
