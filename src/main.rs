extern crate image;

use std::f32;
use std::cmp;
use std::path::Path;
use image::GenericImage;
use std::vec::Vec;

fn open_image(path: &String) -> image::DynamicImage {
    match image::open(&Path::new(&path)) {
        Ok(img) => img,
        Err(err) => {
            println!("{}", err);
            std::process::exit(1);
        }
    }
}

fn compute_gradient(img: &image::DynamicImage) -> Vec<(f32, f32)> {
    let mut grads = vec![(0.0, 0.0); img.width() as usize * img.height() as usize];

    let w = img.width();
    let h = img.height();
    for (x, y, _) in img.pixels() {
        let left = img.get_pixel(cmp::max(0, x as i32 - 1) as u32, y);
        let right = img.get_pixel(cmp::min(x + 1, w - 1), y);
        let up = img.get_pixel(x, cmp::max(0, y as i32 - 1) as u32);
        let down = img.get_pixel(x, cmp::min(y + 1, h - 1));

        let grad_x = (right[0] as i32 + right[1] as i32 + right[2] as i32 -
                      (left[0] as i32 + left[1] as i32 +
                       left[2] as i32) as i32) as f32;
        let grad_y = (down[0] as i32 + down[1] as i32 + down[2] as i32 -
                      (up[0] as i32 + up[1] as i32 + up[2] as i32)) as f32;

        grads[y as usize * w as usize + x as usize] = (grad_x, grad_y);
    }
    return grads;
}

fn pack_cells(grads: &Vec<(f32, f32)>, w: usize, h: usize) -> Vec<(f32, f32)> {
    let mut packed = vec![(0.0, 0.0); (w / 8 + 1) * (h / 8 + 1)];

    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            let mut grad = (0.0, 0.0);
            for miniy in -4i32..4 {
                for minix in -4i32..4 {
                    let this_y = cmp::min(cmp::max(0, y as i32 + miniy), h as i32 - 1) as u32;
                    let this_x = cmp::min(cmp::max(0, x as i32 + minix), w as i32 - 1) as u32;
                    let here = grads[this_y as usize * w + this_x as usize];
                    grad.0 += here.0;
                    grad.1 += here.1;
                }
            }
            packed[(y / 8) as usize * (w / 8) as usize + (x / 8) as usize] = grad;
            x += 8;
        }
        y += 8;
    }
    return packed;
}

fn render_grad(grads: &Vec<(f32, f32)>, w: u32, h: u32) {
    let mut y = 0;
    while y < h / 8 {
        let mut x = 0;
        while x < w / 8 {
            let pi = f32::consts::PI;
            let grad = grads[y as usize * (w / 8) as usize + x as usize];
            let angle = (grad.1.atan2(grad.0) + pi).to_degrees();
            match (angle / 45.0).floor() as i32 {
                0 => print!("→"),
                1 => print!("↘"),
                2 => print!("↓"),
                3 => print!("↙"),
                4 => print!("←"),
                5 => print!("↖"),
                6 => print!("↑"),
                7 => print!("↗"),
                8 => print!("→"),
                _ => print!("💩"),
            }
            x += 1;
        }
        print!("\n");
        y += 1;
    }
}

fn main() {
    let mut args = std::env::args();

    let img = match args.nth(1) {
        Some(path) => open_image(&path),
        None => {
            println!("no path");
            std::process::exit(1);
        }
    };

    println!("computing kernel");
    let grads = compute_gradient(&img);

    println!("packing...");
    let w = img.width();
    let h = img.height();
    let packed = pack_cells(&grads, w as usize, h as usize);

    println!("disp...");
    render_grad(&packed, w, h);
}
