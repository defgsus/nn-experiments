"""
S3M audio renderer

based on 
    - https://github.com/electronoora/webaudio-mod-player

"The basics" are implemented, no effect handling yet

and, unfortunately, it is suuper slow, not usable for now.
It is actually so slow that i did not even test if it's doing
something closely related to rendering the S3M..

"""
import dataclasses
import math
from typing import Optional

import numpy as np

from .s3m import S3m, PatternType


class S3mRenderer:
    
    PERIOD_TABLE = np.array([
        27392.0, 25856.0, 24384.0, 23040.0, 21696.0, 20480.0, 19328.0, 18240.0, 17216.0, 16256.0, 15360.0, 14496.0, 13696.0,
        12928.0, 12192.0, 11520.0, 10848.0, 10240.0, 9664.0, 9120.0, 8608.0, 8128.0, 7680.0, 7248.0, 6848.0, 6464.0, 6096.0,
        5760.0, 5424.0, 5120.0, 4832.0, 4560.0, 4304.0, 4064.0, 3840.0, 3624.0, 3424.0, 3232.0, 3048.0, 2880.0, 2712.0,
        2560.0, 2416.0, 2280.0, 2152.0, 2032.0, 1920.0, 1812.0, 1712.0, 1616.0, 1524.0, 1440.0, 1356.0, 1280.0, 1208.0,
        1140.0, 1076.0, 1016.0, 960.0, 906.0, 856.0, 808.0, 762.0, 720.0, 678.0, 640.0, 604.0, 570.0, 538.0, 508.0, 480.0,
        453.0, 428.0, 404.0, 381.0, 360.0, 339.0, 320.0, 302.0, 285.0, 269.0, 254.0, 240.0, 226.0, 214.0, 202.0, 190.0,
        180.0, 170.0, 160.0, 151.0, 143.0, 135.0, 127.0, 120.0, 113.0, 107.0, 101.0, 95.0, 90.0, 85.0, 80.0, 75.0, 71.0,
        67.0, 63.0, 60.0, 56.0,
    ])
    
    RETRIG_VOL_TAB = np.array([0, -1, -2, -4, -8, -16, 0.66, 0.5, 0, 1, 2, 4, 8, 16, 1.5, 2.0])
    
    _VIBRATO_TABLE: Optional[np.ndarray] = None

    @dataclasses.dataclass
    class Channel:
        index: int
        sample: int = 0
        note: int = 24
        command: int = 0
        data: int = 0
        samplepos: float = 0.
        samplespeed: int = 0
        flags: int = 0
        noteon: int = 0
    
        slidespeed: int = 0
        slideto: int = 0
        slidetospeed: int = 0
        arpeggio: int = 0
    
        period: int = 0
        volume: int = 64
        voiceperiod: int = 0
        voicevolume: int = 0
        oldvoicevolume: int = 0
    
        semitone: int = 12
        vibratospeed: int = 0
        vibratodepth: int = 0
        vibratopos: int = 0
        vibratowave: int = 0
    
        lastoffset: int = 0
        lastretrig: int = 0
    
        volramp: int = 0
        volrampfrom: int = 0
    
        trigramp: int = 0
        trigrampfrom: float = 0.0
    
        currentsample: float = 0.0
        lastsample: float = 0.0
    
    def __init__(
            self, 
            module: S3m,
            samplerate: int = 44100,
    ):
        self.module = module
        self.samplerate = samplerate

        self.tick = -1
        self.position = 0
        self.row = 0
        self.flags = 0
        
        self.volume = self.module.header.globalvolume
        self.speed = self.module.header.initialspeed
        self.bpm = self.module.header.initialtempo
        self.stt = 0
        self.breakrow = 0
        self.looprow = 0
        self.loopstart = 0
        self.loopcount = 0
        self.patterndelay = 0
        self.patternwait = 0
        self.patternjump = 0
        self.endofsong = False

        self.channels = [self.Channel(index=i) for i in range(len(self.module.header.channelsettings))]

    def reset(self):
        self.tick = -1
        self.position = 0
        self.row = 0
        self.flags = 0
        self.stt = 0

        self.breakrow = 0
        self.looprow = 0
        self.loopstart = 0
        self.loopcount = 0
        self.patterndelay = 0
        self.patternwait = 0
        self.patternjump = 0
        self.endofsong = False

        self.channels = [self.Channel(index=i) for i in range(len(self.module.header.channelsettings))]

    def current_pattern(self) -> Optional[PatternType]:
        if self.position >= len(self.module.header.orderlist):
            return None
        idx = self.module.header.orderlist[self.position]
        if idx == 255:
            return None
        return self.module.patterns[idx]

    def _next_tick(self):
        # samples-to-tick
        self.stt = (self.samplerate * 60) / self.bpm / 4 / 6

        self.tick += 1
        self.flags |= 1

        if self.tick >= self.speed:
            # this.onNextRow();
            if self.patterndelay:
                if self.tick < (self.patternwait + 1) * self.speed:
                    self.patternwait += 1
                else:
                    self.row += 1
                    self.tick = 0
                    self.flags |= 2
                    self.patterndelay = 0
            else:
                if self.flags & (16 + 32 + 64):
                    if self.flags & 64:
                        # loop pattern?
                        self.row = self.looprow
                        self.flags &= 0xa1
                        self.flags |= 2
                    elif self.flags & 16:
                        # pattern jump/break?
                        self.position = self.patternjump
                        self.row = self.breakrow
                        self.patternjump = 0
                        self.breakrow = 0
                        self.flags &= 0xe1
                        self.flags |= 2

                    self.tick = 0

                else:
                    self.row += 1
                    self.tick = 0
                    self.flags |= 2

        pattern = self.current_pattern()

        # step to new pattern?
        if self.row >= len(pattern):
            self.position += 1
            self.row = 0
            self.flags |= 4
            # skip markers
            while self.module.header.orderlist[self.position] == 254:
                self.position += 1

        if self.position >= self.module.header.numorders or self.module.header.orderlist[self.position] == 255:
            self.endofsong = True

    def _process_tick(self):
        """
        advance player and all channels by a tick
        """
        self._next_tick()

        pattern = self.current_pattern()

        for ch_idx, ch in enumerate(self.channels):
            ch.oldvoicevolume = ch.voicevolume

            if self.flags & 2 and pattern:  # new row
                # print(ch_idx, self.row, f"{len(pattern)}x{len(pattern[ch_idx])}")
                ch.command = pattern[self.row][ch_idx].effect
                ch.data = pattern[self.row][ch_idx].param

                if not (ch.command == 0x13 and (ch.data % 0xf0) == 0xd0):
                    self._process_note(ch)

            # kill empty samples
            if not self.module.instruments[ch.sample].length:
                ch.noteon = 0

            # run effects on each new tick
            if ch.command < 27:
                if self.tick == 0:
                    pass #self.effects_t0[ch.command](ch)
                else:
                    pass #self.effects_t1[ch.command](ch)

            # advance vibrato pos
            ch.vibratopos = (ch.vibratopos + ch.vibratospeed * 2) % 0xff

            if ch.oldvoicevolume != ch.voicevolume:
                ch.volrampfrom = ch.oldvoicevolume
                ch.volramp = 0.

            # recalc sample speed if voiceperiod has changed
            if (ch.flags & 1 or ch.flags & 2) and ch.voiceperiod:
                ch.samplespeed = 14317056. / ch.voiceperiod / self.samplerate

            # clear channel flags
            ch.flags = 0

        # clear global flags after all channels are processed
        self.flags &= 0x70

    def _process_note(self, channel: Channel):
        pattern = self.current_pattern()
        if not pattern:
            return

        row = pattern[self.row][channel.index]

        if row.instrument and self.module.instruments[row.instrument - 1].type:
            inst = self.module.instruments[row.instrument - 1]
            channel.sample = row.instrument - 1
            channel.volume = channel.voicevolume= inst.volume
            if row.note == 255 and channel.samplepos >= inst.length:
                channel.trigramp = 0.
                channel.trigrampfrom = channel.currentsample
                channel.samplepos = 0

        if row.note < 254:
            inst = self.module.instruments[channel.sample]

            # calc period for note
            pidx = (row.note & 0x0f) + (row.note >> 4) * 12
            pv = (8363. * self.PERIOD_TABLE[pidx]) / inst.c2spd

            # noteon, except if command=0x07 ('G') (porta to note) or 0x0c ('L') (porta+volslide)
            if row.effect != 0x07 and row.effect != 0x0c:
                channel.note = row.note
                channel.period = channel.voiceperiod = pv
                channel.samplepos = 0
                if channel.vibratowave > 3:
                    channel.vibratopos = 0

                channel.trigramp = 0.
                channel.trigrampfrom = channel.currentsample

                channel.flags |= 3  # force sample speed recalc
                channel.noteon = 1

            # in either case, set the slide to note target to note period
            channel.slideto = pv

        elif row.note == 254:
            channel.noteon = 0
            channel.voicevolume = 0

        if row.volume <= 64:
            channel.volume = channel.voicevolume = row.volume

    def process(self, buflen: int) -> np.ndarray:
        if self.endofsong:
            return np.zeros((2, buflen))

        buffer = np.zeros((2, buflen))

        for buf_idx in range(buflen):

            if self.stt <= 0:
                self._process_tick()

            output_r, output_l = 0., 0.

            for ch in self.channels:
                fl, fr, fs = 0., 0., 0.

                ch.currentsample = 0.0  # assume note is off
                if ch.noteon or (not ch.noteon and ch.volramp < 1.):
                    inst = self.module.instruments[ch.sample]

                    if ch.samplepos < inst.length:
                        fl = ch.lastsample
                        samplepos_int = math.floor(ch.samplepos)
                        f = ch.samplepos - samplepos_int
                        fs = inst.sample_data[samplepos_int]
                        fl = f * fs + (1.0 - f) * fl

                        # smooth out discontinuities from retrig and sample offset
                        f = ch.trigramp
                        fl = f * fl + (1.0 - f) * ch.trigrampfrom
                        f += 1.0 / 128.0
                        ch.trigramp = min(1.0, f)
                        ch.currentsample = fl

                        # ramp volume changes over 64 samples to avoid clicks
                        fr = fl * (ch.voicevolume / 64.0)
                        f = ch.volramp
                        fl = f * fr + (1.0 - f) * (fl * (ch.volrampfrom / 64.0))
                        f += 1.0 / 64.0
                        ch.volramp = min(1.0, f)

                        # pan samples
                        fr = fl #* mod.pan_r[ch]
                        #fl *= mod.pan_l[ch]

                        output_l += fl
                        output_r += fr

                    oldpos = ch.samplepos
                    ch.samplepos += ch.samplespeed
                    if math.floor(ch.samplepos) > math.floor(oldpos):
                        ch.lastsample = fs

                    if inst.loop:
                        if ch.samplepos >= inst.loop_end:
                            ch.samplepos -= inst.loop_length
                            ch.lastsample = ch.currentsample
                    else:
                        if ch.samplepos >= inst.length:
                            ch.noteon = 0

            buffer[0, buf_idx] = output_l
            buffer[1, buf_idx] = output_r

            self.stt -= 1

        return buffer