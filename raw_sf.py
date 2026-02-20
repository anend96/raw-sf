import os
import tqdm
import logging
import numpy as np
import fitsio
from multiprocessing import Pool

from obsinfo import *
from redigitize import Redigitize

logger = logging.getLogger(__name__)

def get_args():
    import argparse
    agp = argparse.ArgumentParser("raw2sf", description="Converts uGMRT raw to search-mode PSRFITS", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add('-c,--nchan', help='Number of channels', type=int, required=True, dest='nchans')
    add('-s,--sb', help='Sideband', choices=['l', 'u', 'lower', 'upper'], dest='sideband', required=True)
    add('-f,--feed', help='Feed', choices=['cir', 'lin', 'c', 'l'], dest='feed', required=True)
    add('--gulp', help='Samples in a block', dest='gulp', default=2048, type=int)
    add('--beam-size', help='Beam size in arcsec', dest='beam_size', default=4, type=float)
    add('-O', '--outdir', help='Output directory', default="./")
    add('-d', '--debug', action='store_const', const=logging.DEBUG, dest='loglevel')
    add('-v', '--verbose', action='store_const', const=logging.INFO, dest='loglevel')
    add('raw', help='Path to raw file')
    return agp.parse_args()

def process_chunk(chunk_index, fb, GULP, rdi):
    pkg = fb[chunk_index:(chunk_index + GULP)]
    rdi(pkg)
    return rdi.dat[:], rdi.dat_scl.ravel(), rdi.dat_offs.ravel()

if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.loglevel, format="%(asctime)s %(levelname)s %(message)s")
    
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    
    LSB = False
    USB = False
    if args.sideband == 'l' or args.sideband == 'lower':
        LSB = True
    else:
        USB = True
    
    CIRC = False
    LIN = False
    feed = ''
    if args.feed == 'c' or args.feed == 'cir':
        CIRC = True
        feed = 'CIRC'
    else:
        LIN = True
        feed = 'LIN'
    
    GULP = args.gulp
    nch = args.nchans
    npl = 4
    
    raw = args.raw
    baw = os.path.basename(raw)
    hdr = raw + ".hdr"
    logging.info(f"Raw file = {raw}")
    logging.info(f"Raw header file = {hdr}")
    
    rawt = read_hdr(hdr)
    logging.info(f"Raw MJD = {rawt.mjd:.5f}")
    
    src = baw.split('_')[0].upper()
    logging.info(f"Source = {src}")
    if src not in MISC_SOURCES:
        raise ValueError("source not found")
    
    rfb = np.memmap(raw, dtype=np.int16, mode='r', offset=0)
    fb = rfb.reshape((-1, nch, npl))
    logging.debug(f"Raw shape = {fb.shape}")
    
    band = get_band(baw)
    tsamp = get_tsamp(band, nch)
    freqs = get_freqs(band, nch, lsb=LSB, usb=USB)
    logging.info(f"Raw band = {band}")
    logging.debug(f"Tsamp = {tsamp}")
    logging.debug(f"Frequencies = {freqs[0]:.3f} ... {freqs[-1]:.3f}")
    
    rdi = Redigitize(GULP, nch, npl, feed)
    
    nsamples = fb.shape[0]
    nrows = nsamples // GULP
    fsamples = nrows * GULP
    last_row = nsamples - fsamples
    logging.debug(f"Total samples = {nsamples:d}")
    logging.debug(f"Number of rows = {nrows:d}")
    logging.debug(f"Samples in last row = {last_row:d}")
    
    row_size = GULP * nch * npl * 2
    
    tr = tqdm.tqdm(range(0, fsamples, GULP), desc='raw2sf', unit='blk')
    
    outfile = os.path.join(args.outdir, baw + ".sf")
    logging.info(f"Output search-mode psrfits = {outfile}")
    
    d = BaseObsInfo(rawt.mjd, 'search', circular=CIRC, linear=LIN)
    d.fill_freq_info(nch, band['bw'], freqs)
    d.fill_source_info(src, RAD[src], DECD[src])
    d.fill_beam_info(args.beam_size)
    d.fill_data_info(tsamp)
    
    t_row = GULP * tsamp
    tsubint = np.ones(nrows, dtype=np.float64) * t_row
    offs_sub = (np.arange(nrows) + 0.5) * t_row
    lst_sub = d.get_lst_sub(offs_sub)
    
    ra_deg, dec_deg = d.sc.ra.degree, d.sc.dec.degree
    scg = d.sc.galactic
    l_deg, b_deg = scg.l.value, scg.b.value
    ra_sub = np.ones(nrows, dtype=np.float64) * ra_deg
    dec_sub = np.ones(nrows, dtype=np.float64) * dec_deg
    glon_sub = np.ones(nrows, dtype=np.float64) * l_deg
    glat_sub = np.ones(nrows, dtype=np.float64) * b_deg
    fd_ang = np.zeros(nrows, dtype=np.float32)
    pos_ang = np.zeros(nrows, dtype=np.float32)
    par_ang = np.zeros(nrows, dtype=np.float32)
    tel_az = np.zeros(nrows, dtype=np.float32)
    tel_zen = np.zeros(nrows, dtype=np.float32)
    dat_freq = np.vstack([freqs] * nrows).astype(np.float32)
    dat_wts = np.ones((nrows, nch), dtype=np.float32)
    dat_wts[:, :50] = 0
    dat_wts[:, -10:] = 0
    dat_offs = np.zeros((nrows, nch, npl), dtype=np.float32)
    dat_scl = np.ones((nrows, nch, npl), dtype=np.float32)
    dat = np.array([], dtype=np.uint8)
    
    phdr = d.fill_primary_header(chan_dm=0., scan_len=t_row * nrows)
    subinthdr = d.fill_search_table_header(GULP)
    fits_data = fitsio.FITS(outfile, 'rw', clobber=True)
    
    fits_data.write(None, header=phdr, extname='PRIMARY')
    
    subint_columns = {
        "TSUBINT": tsubint,
        "OFFS_SUB": offs_sub,
        "LST_SUB": lst_sub,
        "RA_SUB": ra_sub,
        "DEC_SUB": dec_sub,
        "GLON_SUB": glon_sub,
        "GLAT_SUB": glat_sub,
        "FD_ANG": fd_ang,
        "POS_ANG": pos_ang,
        "PAR_ANG": par_ang,
        "TEL_AZ": tel_az,
        "TEL_ZEN": tel_zen,
        "DAT_FREQ": dat_freq,
        "DAT_WTS": dat_wts,
        "DAT_OFFS": dat_offs,
        "DAT_SCL": dat_scl,
        "DATA": dat
    }
    
    for key, value in subint_columns.items():
        fits_data.write(value, header=subinthdr, extname='SUBINT')
    
    isubint = 0
    with Pool(processes=4) as pool:  # Adjust the number of processes as needed
        results = pool.starmap(process_chunk, [(i, fb, GULP, Redigitize(GULP, nch, npl, feed)) for i in range(0, fsamples, GULP)])
        for result in results:
            dat, dat_scl, dat_offs = result
            subint_sf = fits_data['SUBINT']
            subint_sf.write(dat, row=isubint)
            subint_sf.write(dat_scl, row=isubint)
            subint_sf.write(dat_offs, row=isubint)
            isubint += 1
            fits_data.flush()
    
    fits_data.close()
