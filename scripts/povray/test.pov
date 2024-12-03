camera
{
    orthographic
    location <0, 0, -3>
    look_at <0, 0, 0>
    right x
}

light_source { <-2,2,-3> rgb 1 }


#declare TEXTURE = texture
{
	pigment { rgb .5 }
	//normal { average normal_map { [nOrm (0.6,0.0008)] [nOrm (0.1,0.23)] } }
	finish {
		ambient .2 diffuse .5 brilliance 0.8
		phong 1. phong_size 10.
    }
}


cylinder {
    0, x * 1, .05
    texture { TEXTURE }
}