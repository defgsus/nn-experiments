

class WangTiles2:

    L = 1
    T = 1 << 1
    R = 1 << 2
    B = 1 << 3

    TL = 1 << 4
    TR = 1 << 5
    BR = 1 << 6
    BL = 1 << 7

    ALL_EDGES = L|R|T|B
    ALL_CORNERS = TL|TR|BL|BR


    LAYOUTS = {
        "edge-4x4": [
            [0, R, L|R, L],
            [B, B|R, B|L|R, B|L],
            [B|T, B|R|T, B|L|R|T, B|L|T],
            [T, R|T, L|R|T, L|T],
        ],
        "corner-4x4": [
            [0, R, L|R, L],
            [B, B|R, B|L|R, B|L],
            [B|T, B|R|T, B|L|R|T, B|L|T],
            [T, R|T, L|R|T, L|T],
        ],
        "edge-corner-7x7": [
            [B|BL|BR|L|R|T|TL|TR, B|BL|BR|L|T|TL|TR, BL|T|TL|TR, T|TL|TR, BR|T|TL|TR, BL|BR|R|T|TL|TR, B|BL|BR|L|R|T|TL|TR],
            [BL|BR|L|R|T|TL|TR, BL|BR|L|T|TL|TR, BL|BR|TL, BL, BR|R|TR, BL|BR|L|TL|TR, B|BL|BR|R|T|TL|TR],
            [BL|L|TL|TR, BR|TL|TR, BL|BR|TL|TR, B|BL|BR|TL, BL|TR, TL|TR, BR|R|T|TL|TR],
            [BL|L|TL, TR, BR|R|TL|TR, BL|L|T|TL|TR, TL, BR, BL|BR|R|TR],
            [BL|BR|L|TL, B|BL|BR, BL|BR|TR, BL|TL, 0, BR|TR, BL|BR|R|TL|TR],
            [B|BL|BR|L|TL|TR, BL|BR|T|TL|TR, BL|TL|TR, BR|TL, BL|BR, B|BL|BR|R|TR, BL|BR|L|R|TL|TR],
            [B|BL|BR|L|R|T|TL|TR, B|BL|BR|L|R|TL|TR, B|BL|BR|L|TL, B|BL|BR|TR, B|BL|BR|TL|TR, B|BL|BR|T|TL|TR, B|BL|BR|R|TL|TR],
        ],
    }