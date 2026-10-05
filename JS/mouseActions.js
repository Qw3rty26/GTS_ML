let initialMouseX = 0;
let initialMouseY = 0;
let isDragging = false;

systemCanvas.addEventListener("mousedown", (event)=>{ // used to drag the space
        //TODO Implement camera following an entity
        event.preventDefault(); // remove default handlers
        isDragging = true;
        initialMouseX = event.clientX - rect.left;
        initialMouseY = event.clientY - rect.top;
})

systemCanvas.addEventListener("mousemove", (event)=>{ // used to drag the space
        event.preventDefault(); // remove default handlers
        if(isDragging){
                let finalMouseX = event.clientX - rect.left;
                let finalMouseY = event.clientY - rect.top;

                let draggedX = finalMouseX - initialMouseX;
                let draggedY = finalMouseY - initialMouseY;

                entitySystem.space.dragSpace(gridCtx, draggedX, draggedY); // redraw the grid
                entitySystem.renderEntities(systemCtx); // render the system

                initialMouseX = finalMouseX;
                initialMouseY = finalMouseY;
        }
})

systemCanvas.addEventListener("mouseup", (event)=>{ // used to drag the space
        isDragging = false;
})

systemCanvas.addEventListener("wheel", (event)=>{ // used to zoom in and out
        event.preventDefault(); // remove default handlers
        const zoomAmount = event.deltaY > 0 ? -5 : 5;
        entitySystem.space.zoomSpace(gridCtx, zoomAmount); // redraw the grid
        entitySystem.renderEntities(systemCtx); // render the system
})

window.addEventListener("resize", (event)=>{
        winWidth = window.innerWidth;
        winHeight = window.innerHeight;

        entitySystem.space.screenWidth = window.innerWidth;
        entitySystem.space.screenHeight = window.innerHeight;

        setCanvasSize(gridCanvas, winWidth, winHeight);
        setCanvasSize(systemCanvas, winWidth, winHeight);

        entitySystem.space.dragSpace(gridCtx, 0, 0); // redraw the grid
        entitySystem.renderEntities(systemCtx); // render system
})
