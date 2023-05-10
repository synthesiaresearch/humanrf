function imageZoom(imgID, resultID, initial_offset_x=0, initial_offset_y=0) {
    // based on https://www.w3schools.com/howto/howto_js_image_zoom.asp
    var img, lens, result, cx, cy;
    img = document.getElementById(imgID);
    result = document.getElementById(resultID);
    /* Create lens: */
    lens = document.createElement("DIV");
    lens.setAttribute("class", "img-zoom-lens");
    /* Insert lens: */
    img.parentElement.insertBefore(lens, img);
    /* Calculate the ratio between result DIV and lens: */
    cx = result.offsetWidth / lens.offsetWidth;
    cy = result.offsetHeight / lens.offsetHeight;
    /* Set background properties for the result DIV */
    result.style.backgroundImage = "url('" + img.src + "')";
    result.style.backgroundSize = (img.width * cx) + "px " + (img.height * cy) + "px";
    /* Execute a function when someone moves the cursor over the image, or the lens: */
    lens.addEventListener("mousemove", move_lens);
    img.addEventListener("mousemove", move_lens);
    /* And also for touch screens: */
    lens.addEventListener("touchmove", move_lens);
    img.addEventListener("touchmove", move_lens);
    function set_lens(posx, posy) {
        /* Calculate the position of the lens: */
        x = posx - (lens.offsetWidth / 2);
        y = posy - (lens.offsetHeight / 2);
        /* Prevent the lens from being positioned outside the image: */
        if (x > img.width - lens.offsetWidth) {x = img.width - lens.offsetWidth;}
        if (x < 0) {x = 0;}
        if (y > img.height - lens.offsetHeight) {y = img.height - lens.offsetHeight;}
        if (y < 0) {y = 0;}
        /* Set the position of the lens: */
        lens.style.left = x + "px";
        lens.style.top = y + "px";
        /* Display what the lens "sees": */
        result.style.backgroundPosition = "-" + (x * cx) + "px -" + (y * cy) + "px";
    }
    function move_lens(e) {
        var pos, x, y;
        /* Prevent any other actions that may occur when moving over the image */
        e.preventDefault();
        /* Get the cursor's x and y positions: */
        pos = get_cursor_pos(e);
        set_lens(pos.x, pos.y);
    }
    function get_cursor_pos(e) {
        var a, x = 0, y = 0;
        e = e || window.event;
        /* Get the x and y positions of the image: */
        a = img.getBoundingClientRect();
        /* Calculate the cursor's x and y coordinates, relative to the image: */
        x = e.pageX - a.left;
        y = e.pageY - a.top;
        /* Consider any page scrolling: */
        x = x - window.pageXOffset;
        y = y - window.pageYOffset;
        return {x : x, y : y};
    }

    if (initial_offset_x > 0 || initial_offset_y > 0) {
        set_lens(initial_offset_x, initial_offset_y);
    }
}