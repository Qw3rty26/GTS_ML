const COLOR_WHITE = "#E6EAF2";

class Entity{
	constructor(id = 0, xPos = 0.0, yPos = 0.0, zPos = 0.0, xVel = 0.0, yVel = 0.0, zVel = 0.0, mass = 0.0){
		this.id = id;
                this.xPos = xPos;
		this.yPos = yPos;
                this.zPos = zPos;
		this.xVel = xVel;
		this.yVel = yVel;
                this.zVel = zVel;
		this.mass = mass;
		this.radius = 90;
	}

	render(ctx, space, oldE, alpha) {
        	//TODO make it so that whenever the entity is outside the screen, it wont get rendered
		const newRadius = this.radius * space.scale;

		let interpolatedX = this.xPos;
        	let interpolatedY = this.yPos;

        	if (oldE && alpha !== undefined) {
                	interpolatedX = oldE.xPos + (this.xPos - oldE.xPos) * alpha;
                	interpolatedY = oldE.yPos + (this.yPos - oldE.yPos) * alpha;
        	}

		const {x, y} = space.toScreen(interpolatedX, interpolatedY);

		ctx.save();
		ctx.beginPath();
		ctx.arc(x, y, newRadius, 0, 2 * Math.PI); // draw circonference;
		ctx.fillStyle = COLOR_WHITE;
        	ctx.strokeStyle = COLOR_WHITE;
		ctx.fill();
		ctx.restore();
	};

	//TODO: TO BE FIXED
	renderMass(ctx, space) {
    		const { x, y } = space.toScreen(this.xPos, this.yPos);
    		const radius = this.radius * space.scale;

    		ctx.save();
    		ctx.fillStyle = COLOR_WHITE;
    		ctx.font = "12px Arial";
    		ctx.textAlign = "center";
    		ctx.textBaseline = "middle";
    		ctx.fillText(this.mass, x, y);
    		ctx.restore();
	}

	renderVelocityVector(ctx, space) {
		const {x, y} = space.toScreen(this.xPos, this.yPos);
		ctx.save();
		ctx.beginPath();
		ctx.moveTo(x, y);
		ctx.lineTo(x + this.xVel * space.scale * 50, y + this.yVel * space.scale * 50);
		ctx.strokeStyle = "red";
		ctx.lineWidth = 2;
		ctx.stroke();
                ctx.restore();
	}

}
