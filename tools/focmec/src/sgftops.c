/*
 * ==========================================================================
 * PURPOSE:  To convert SAC Graphics Files to a Postscript File.
 * =========================================================================
 * MODULE/LEVEL:  sgfcodes/1
 * =========================================================================
 * RELATED ROUTINES:
 *   sgftoeps.csh:  Script (using gs) produces encapsulated PS file.
 *   sgftox.csh:    Script (using gs) produces screen plot.
 * =========================================================================
 * MODIFICATION HISTORY:
 *  201305:   Arthur and Brian changed base_width from 15 to 40 to maatch
              output linewidths from savimg ps or pdf files.
 *  201206:               changed linestyles to match aux/linestyles.txt
 *  201205:               cleaned up code
 *  20100614:  rwg@vt.edu fixed a bug related to the "move" command (values
 *             of x and y were not saved between calls to execute_buffer)
 *  20100208:  snoke@vt.edu and rwg@vt.edu made program independent of byte
 *             order of .sgf file
 *  20090912:  snoke@vt.edu moved setlinewdith from prologue. Fixed(>)
 *             problem with some print drivers that did not see it.
 *  20090505:  snoke@vt.edu added endian check.  If inconsistent, exits
 *  20090311:  rwg@vt.edu prevented the output of redundant "move" and
 *             "setrgbcolor" commands; other cleanups
 *    060819:  snoke@vt.edu tweaked formatting
 *    060325:  rwg@vt.edu changed all gets() calls to fgets() and
 *             modified code to get rid of most warnings from splint
 *    060221:  Corrected the BoundingBox -- wrong order.
 *    050104:  Following advice of Russell Lang, author of epstool, jas
 *             changed the syntax in the header output lines.
 *    030924:  rwg@vt.edu changed username detection code to not use the
 *             getlogin() function
 *    030627:  rwg@vt.edu tried to bring code into ANSI C/POSIX world
 *             time/date in id label now file creation time/date
 *    030531:  jas/vt changed first line output to claim EPSF file
 *    001013   jas/vt changed %%page: ?? to %%Pages: 1 so epstool works
 *    960228:  jas/vtso added in scaling, rotating, labeling, linewidth
 *             features he & Edmundo Norabuena introduced for sgftops. 
 *             Changes they made are marked below with a *
 *    920908:* ENORAB fix minor bug appearing when using argc = 4.
 *    920907:* JASnoke added id=-9 as switch to change line thicknesses
 *    920812:  Merged filled rectangles/ colortables: Stephanie Daveler.
 *    920811:  Merged filled polygons: Chuck A & Myers (U of Arizona)
 *    920522:  Added line-width opcode (-9) command.
 *    920705:* JAS (1) changed font type and placement in label option
 *              (2) fine-tuned linewidth change option allow noninteger
 *                  multiples of 15 (linewidth = 1) on input
 *              (3) Corrected error in scaling option
 *    920403:* ENorab(VT), option for translate/rotate/scaling
 *             in order to get portrait mode plots. 
 *             Arguments are limited to 5
 *             and after line thickness possible new options are:
 *              (i) put id on standard landscape.
 *              (s) query for translate/rotate/scaling parameters.
 *              (si)query scaling parameters and put id to the plot.
 *    910309:  Changed "%%Page: ? 1" to "%%Page: ? ?" no duplicate page
 *             problem.
 *    910128:  Added "%%Page: ? 1" - enables pageview to preview .ps
 *    900710:  Added flag on command line to ignore scaling.
 *    900709:  Added prompt for scaling when scale parameter is 32000.
 *    900323:  Increased available linestyles from 5 to 10.
 *             Added mod 10 to line style selection.
 *    900312:  Updated op-code tests to match fortran version.
 *             Added new 10" width scaling factor and plotsize opt.
 *    870921:  Added optional linewidth and plot id label plus bug fixes
 *    870502:  Made it more robust and more readable
 *    870113:  Modified input arguments and some logic.
 *    860000:  Original version due to Tom Owens at Univ. of Missouri.
 * =========================================================================
 */

#include <fcntl.h>
#include <sys/types.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define TRUE 1
#define FALSE 0
#define maxbuflen 5004
#define NUMLSTYLS 10 /* 10 line styles */
#define NUMLWIDTHS 10 /* 10 line widths */
#define NUMCOLORS 256 /* 256 color table entries */

/*
 * Available Line Styles:
 * styles are defined later by:  "[bu wu bu wu] offset"
 * where bu=black units, wu=white units, offset=0.
 * Numbers are large because of reduced scaling (.0225)
 * of the sgf file (32000 X 2400) onto 10" X 7.5" plot.
 * 
 * style  description                      PS setdash definition
 * -----  -------------------------------  ---------------------
 * ls1:   solid                            [] 0
 * ls3:   sm dashes                        [200 150] 0
 * ls4:   lg dashes                        [600 150] 0
 * ls5:   lg dash, dot                     [600 150 75 150] 0
 * ls6:   lg dash, sm dash                 [600 200 200 200] 0
 * ls7:   lg dash, dot, dot                [600 150 75 150 75 150] 0
 * ls8:   lg dash, sm dash, sm dash        [600 150 200 150 200 150] 0
 * ls2:   dots                             [75 75] 0
 * ls9:   lg dash, dot, dot, dot           [600 150 75 150 75 150 75 150] 0
 * ls10:  lg, sm, sm, sm dashes            [600 150 200 150 200 150 200 150] 0
 */

static const char *linemodes[] = {
    "ls1", "ls2", "ls3", "ls4", "ls5",
    "ls6", "ls7", "ls8", "ls9", "ls10"
};

static char scale[5] = "yes"; /* for prompt or command line parameter */
static int scale_flag = TRUE;

/* #define COLORTABLE  */
static int textangle = 0;
/* static int colortable = FALSE; */
/* static char *colorfile = "colortable"; */

static int imagecount = 0;
static int icount = 0;
static int doing_image = FALSE;

static char xshif[10] = "0", yshif[10] = "0";
static char sfact[10] = "1";
static char angle[10] = "0";
static int base_width;

static int same = TRUE;
static int first = TRUE;

/* PROTOTYPES --------------------------------------------------------------- */

static void execute_buffer(short *buffer, int buflen, int *done, FILE *ofp,
                           float red[], float green[], float blue[], int ict);
static void EndPath(FILE *ofp, int *fillFlag);

static int32_t int_swap32(char cbuf[]);
static int32_t int_join32(char cbuf[]);
static int16_t int_swap16(char cbuf[]);
static int16_t int_join16(char cbuf[]);

/* -------------------------------------------------------------------------- */

int main(int argc, char *argv[]) {
    char cbuf2[2], cbuf4[4];
    FILE *file_in, *ofp;
    short *buffer;
    size_t nbuffer;
    int j, buflen, done = 0, ict;
    float red[NUMCOLORS], green[NUMCOLORS], blue[NUMCOLORS];
    char usnam[256] = "", path[MAXPATHLEN], *dattim;
    struct passwd *pw;
    double width_in;
    struct stat s;
    
    /* be sure we have enough arguments */
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: %s sgf_file ps_file", argv[0]);
        fprintf(stderr, " [line_width scale_id]\n");
        fprintf(stderr, " \n");
        fprintf(stderr, "where: line_width = 1, 1.5, 2, 3, etc.\n");
        fprintf(stderr, "       scale_id = i  (landscape mode plus id)\n");
        fprintf(stderr, "       scale_id = s  (shift,rotate & scale)\n");
        fprintf(stderr, "       scale_id = si (s plus id)\n");
        fprintf(stderr, "       time/date in id is file creation date\n");
        fprintf(stderr, " \n");
        fprintf(stderr, "example: sgftops foo.sgf foo.ps 2 si\n");
        fprintf(stderr, " \n");
        fprintf(stderr, "Produces a plot with line thick=2 and ID at the "
                "bottom.\n");
        fprintf(stderr, "   Prompts further for translation, rotation and "
                "scale.\n");
        fprintf(stderr, " \n");
        fprintf(stderr, "* origin of plot is lower left corner of portrait "
                "mode and angle is CCW\n");
        exit(-1);
    }
    
    /* does argv[1] end in ".sgf" ? */
    if (strcmp(&argv[1][strlen(argv[1]) - 4], ".sgf") != 0) {
        fprintf(stderr, "%s: %s not a SAC GRAPHICS FILE.\n", *argv, argv[1]);
        exit(1);
    }
    
    /* open the sgf file */
    if ((file_in = fopen(argv[1], "rb")) == NULL) {
        fprintf(stderr, "%s: Can't open %s for reading.\n", *argv, argv[1]);
        exit(-3);
    }
    
    /* open the postscript file to write */
    ofp = fopen(argv[2], "w");
    if (ofp == NULL) {
        fprintf(stderr, "cannot open postscript file to write: '%s'\n",
                argv[2]);
        exit(-2);
    }
    
    /* initialize the buffer */
    nbuffer = (maxbuflen * sizeof(short));
    buffer = (short *)malloc( nbuffer );
    if (buffer == NULL) {
        fprintf(stderr, "cannot create input buffer\n");
        exit(-4);
    }
    /* initialize buffer contents to nothing */
    memset(buffer, 0, nbuffer);

    /* get the info for the plot id label */
    pw = getpwuid(geteuid());
    if (pw != NULL && pw->pw_name != NULL) {
        strncpy(usnam, pw->pw_name, sizeof(usnam));
    }
    
    (void)getcwd(path, (size_t)MAXPATHLEN);
    
    if (fstat(fileno(file_in), &s) != 0) {
        fprintf(stderr, "cannot stat input file\n");
        exit(-5);
    }
    dattim = ctime(&s.st_mtime);
    
    /* define some abreviations for some postscript commands */
    /* and use optional line width if argument present       */
    fprintf(ofp, "%%!PS-Adobe-3.0 EPSF-3.0\n");
    fprintf(ofp, "%%%%BoundingBox: 0 0 612 792\n");
    fprintf(ofp, "%%%%Creator: sgftops (20090313)\n");
    fprintf(ofp, "%%%%Pages: 1\n");
    fprintf(ofp, "%%%%EndComments\n");
    fprintf(ofp, "%%%%BeginProlog\n");
    fprintf(ofp, "/Helvetica findfont 11 scalefont setfont");
    fprintf(ofp, "\n/ls1 {[] 0 setdash} def");
    fprintf(ofp, "\n/ls2 {[200 150] 0 setdash} def");
    fprintf(ofp, "\n/ls3 {[600 150] 0 setdash} def");
    fprintf(ofp, "\n/ls4 {[600 150 75 150] 0 setdash} def");
    fprintf(ofp, "\n/ls5 {[600 150 200 150] 0 setdash} def");
    fprintf(ofp, "\n/ls6 {[600 150 75 150 75 150] 0 setdash} def");
    fprintf(ofp, "\n/ls7 {[600 159 200 150 200 150] 0 setdash} def");
    fprintf(ofp, "\n/ls8 {[75 75] 0 setdash} def");
    fprintf(ofp, "\n/ls9 {[600 150 75 150 75 150 75 150] 0 setdash} def");
    fprintf(ofp, "\n/ls10 {[600 150 200 150 200 150 200 150] 0 setdash} def");
    fprintf(ofp, "\n/L {lineto} def");
    fprintf(ofp, "\n/m {moveto} def");
    fprintf(ofp, "\n/g {setgray} def");
    fprintf(ofp, "\n/s {stroke} def");
    
    /* define procedure to make filled rectangles */
    fprintf(ofp, "\n/F {");
    fprintf(ofp, "\n   /blue exch def");
    fprintf(ofp, "\n   /green exch def");
    fprintf(ofp, "\n   /red exch def");
    fprintf(ofp, "\n   /dy exch def");
    fprintf(ofp, "\n   /dx exch def");
    fprintf(ofp, "\n   /y exch def");
    fprintf(ofp, "\n   /x exch def");
    fprintf(ofp, "\n   s");
    fprintf(ofp, "\n   red green blue setrgbcolor");
    fprintf(ofp, "\n   x y m");
    fprintf(ofp, "\n   0 dy rlineto");
    fprintf(ofp, "\n   dx 0 rlineto");
    fprintf(ofp, "\n   0 dy neg rlineto");
    fprintf(ofp, "\n   closepath");
    fprintf(ofp, "\n   fill");
    fprintf(ofp, "\n} def");
    
    base_width = 40;
    if (argc > 3) {
        width_in = atof(argv[3]);
        
        if (width_in <= 0.0 || width_in > 9.0)
            width_in = 1.0;
        
        base_width = (int)(base_width * width_in);
    }
    fprintf(ofp, "\n");
    fprintf(ofp, "%%%%EndProlog\n");
    fprintf(ofp, "%%%%Page: 1 1\n");
    if (argc > 4 && strncmp(argv[4], "s", 1) == 0) {
        printf("First translates (x and y), then rotates, then scales:\n");
        printf("   [Default] landscape: 8 0 90 1  to prompts\n");
        printf("   Sample portrait:  0.5 0.5 0 0.75\n");
        printf(" \n");
        
        printf("\nx translation : ");
        (void)fgets(xshif, (int)sizeof(xshif), stdin);
        xshif[strlen(xshif) - 1] = '\0'; /* remove newline char */
        
        printf("y translation : ");
        (void)fgets(yshif, (int)sizeof(yshif), stdin);
        yshif[strlen(yshif) - 1] = '\0'; /* remove newline char */
        
        printf("rotation angle: ");
        (void)fgets(angle, (int)sizeof(angle), stdin);
        angle[strlen(angle) - 1] = '\0'; /* remove newline char */
        
        printf("scale........ : ");
        (void)fgets(sfact, (int)sizeof(sfact), stdin);
        sfact[strlen(sfact) - 1] = '\0'; /* remove newline char */
        
        fprintf(ofp, "gsave");
        fprintf(ofp, "\n72 %s mul 72 %s mul translate", xshif, yshif);
        fprintf(ofp, "\n%s rotate", angle);
        fprintf(ofp, "\n10 72 mul 32000 div 7.5 72 mul 24000 div scale");
        fprintf(ofp, "\n%s %s scale", sfact, sfact);
    } else {
        fprintf(ofp, "gsave");
        fprintf(ofp, "\n72 8 mul 36 translate");
        fprintf(ofp, "\n90 rotate");
        fprintf(ofp, "\n10 72 mul 32000 div 7.5 72 mul 24000 div scale");
    }
    
    /* add the optional label for userid, dir, file, date and time */
    if (argc > 4) {
        if (strncmp(argv[4], "i", 1) == 0 || strncmp(argv[4], "si", 2) == 0) {
            fprintf(ofp, "\n20 20 m");
            fprintf(ofp, "\ngsave");
            fprintf(ofp, "\n32000 72 10 mul div 24000 72 7.5 mul div scale");
            fprintf(ofp, "\n(%s          )show", usnam);
            fprintf(ofp, "\n(%s		)show", path);
            fprintf(ofp, "\n360 0 m (%s) stringwidth pop 2 div neg 0 rmoveto",
                    argv[1]);
            fprintf(ofp, "\n(%s)show", argv[1]);
            fprintf(ofp, "\n720 0 m (%s) stringwidth pop neg 0 rmoveto",
                    dattim);
            fprintf(ofp, "\n(%s)show", dattim);
            fprintf(ofp, "\ns");
            fprintf(ofp, "\ngrestore");
        }
    }
    
    /* set the scale flag on or off */
    if (argc > 5 && strncmp(argv[5], "no", 2) == 0)
        scale_flag = FALSE;
	
    fprintf(ofp, "\n%d setlinewidth", base_width);
    
    /* look for colortable file */
    ict = 0;
#ifdef COLORTABLE    
    if (colortable == TRUE) {
        fpc = fopen(colorfile, "r");
        if ((fpc = fopen(colorfile, "r")) == NULL) {
            fprintf(stderr, "%s: %s color table file missing.\n", *argv,
                    colorfile);
            exit(1);
        }
        
        /* user defined colortable */
        while (fscanf(fpc, "%g %g %g", &red[ict], &green[ict], &blue[ict]) == 3
               && ict < NUMCOLORS)
            ict++;
        
        (void)fclose(fpc);
    } else 
#endif /* COLORTABLE */
    {
        /* default colortable */
        red[ict] = 1.0; green[ict] = 1.0; blue[ict] = 1.0; ict++;
        red[ict] = 1.0; green[ict] = 0.0; blue[ict] = 0.0; ict++;
        red[ict] = 0.0; green[ict] = 1.0; blue[ict] = 0.0; ict++;
        red[ict] = 0.0; green[ict] = 0.0; blue[ict] = 1.0; ict++;
        red[ict] = 1.0; green[ict] = 1.0; blue[ict] = 0.0; ict++;
        red[ict] = 0.0; green[ict] = 1.0; blue[ict] = 1.0; ict++;
        red[ict] = 1.0; green[ict] = 0.0; blue[ict] = 1.0; ict++;
        red[ict] = 0.0; green[ict] = 0.0; blue[ict] = 0.0; ict++;
    }
    
    /* execute this loop once for each input record */
    while (done == 0) {
        if (fread(cbuf4, 4, 1, file_in) != 1) {
            done = 1;
            break; /* no more to read */
        }

        if (first) {
            first = FALSE;
	    buflen = int_join32(cbuf4);

            if (buflen >= 65536) {
                buflen = int_swap32(cbuf4);
                same = FALSE;
            }
        } else {
            if (same == FALSE) {
                buflen = int_swap32(cbuf4);
            }
	    else buflen = int_join32(cbuf4); 
        }
        
        buflen = buflen * 2;

        for (j = 0; j < buflen; j++) {
            if (fread(cbuf2, 2, 1, file_in) != 1) {
                fprintf(stderr, "%s: Problem with buffer read for j = %d\n", 
                        *argv, j);
                exit(-1);
            }

            if (same == FALSE) {
                buffer[j] = int_swap16(cbuf2);
            } else {
                buffer[j] = int_join16(cbuf2);
            }
        }

        execute_buffer(buffer, buflen, &done, ofp, red, green, blue, ict);
    } /* while */
    
    (void)fclose(file_in);
    
    /* print the page */
    fprintf(ofp, "\ns");
    fprintf(ofp, "\nshowpage");
    fprintf(ofp, "\ngrestore");
    fprintf(ofp, "\n");
    fprintf(ofp, "%%%%Trailer\n");
    
    (void)fclose(ofp);
    
    if(buffer) {
        free(buffer);
        buffer = NULL;
    }

    return 0;
}

/* ------------------------------------------------------------------- */

static void execute_buffer(short *buffer, int buflen, int *done, FILE *ofp,
                      float red[], float green[], float blue[], int ict) {
    int i, newindx;
    int count, dx, dy;
    int iwidth, iheight, xloc, yloc, bufcount, nwrite, charcount;
    int new_width;
    char str[133];
    
    static int x, y;
    static int indx = -1, needmove = 0;
    
    /* for hardware text size */
    
    float xsize; /* requested horizontal size in " */
    float xscale, yscale;
    
    static int fillFlag = 0;
    int igray;
    unsigned short hexnum;
    unsigned char *cptr;
    float gray_value = 0.;
    
    /* are we in the middle of writing out an image? */
    if (doing_image == TRUE) {
        /* print out rgb values 24 to a line */
        bufcount = 0;
        charcount = 0;
        
        cptr = (unsigned char *)buffer;
        
        while (((icount + 24) <= imagecount)&&((bufcount+12) <= buflen)) {
            for (i = 0; i < 24; i++) {
                hexnum = (unsigned short)cptr[charcount++];
                fprintf(ofp, "%2.2hx", hexnum);
            }
            fprintf(ofp, "\n");
            icount += 24;
            bufcount += 12;
        }
        
        if ((buflen - bufcount) > 0) {
            nwrite = 2 * (buflen - bufcount);
            nwrite = (icount+nwrite) <= imagecount ? nwrite : imagecount -
            icount;
            icount += nwrite;
            for (i = 0; i < nwrite; i++) {
                hexnum = (unsigned short)cptr[charcount++];
                fprintf(ofp, "%2.2hx", hexnum);
            }
            fprintf(ofp, "\n");
        }
        
        if (icount >= imagecount) {
            doing_image = FALSE;
            fprintf(ofp, "\n grestore");
        }
        
        return;
    }
    
    for (i = 0; i < buflen && *done == 0; ) {
        if (needmove != 0 &&
            (buffer[i] >= 0 || buffer[i] == -5 || buffer[i] == -10)) {
            EndPath(ofp, &fillFlag);
            fprintf(ofp, "\n%d %d m", x, y);
            needmove = 0;
        }
        if (buffer[i] >= 0) {
            x = buffer[i++];
            y = buffer[i++];
            fprintf(ofp, "\n%d %d L", x, y);
        } else {
            switch (buffer[i++]) {
                case -1: /* No-op command */
                    break;
                case -2: /* End of plot command */
                    *done = 1;
                    i++;
                    break;
                case -3: /* Move command */
                    i++;
                    x = buffer[i++];
                    y = buffer[i++];
                    needmove++;
                    break;
                case -4: /* Color command - default SAC color map  */
                    i++;
                    newindx = buffer[i++];
                    
                    if (newindx >= 0 && newindx < ict) {
                        if (newindx != indx) {
                            indx = newindx;
                            EndPath(ofp, &fillFlag);
                            fprintf(ofp, "\n%g %g %g setrgbcolor", red[indx],
                                    green[indx], blue[indx]);
                        }
                    } else {
                        EndPath(ofp, &fillFlag);
                        fprintf(ofp, "\n0.0 0.0 0.0 setrgbcolor");
                        printf("Illegal color value encountered - set to "
                               "default color black\n");
                    }
                    break;
                case -5: /* Hardware text command */
                    i++;
                    i += 5;  /* modified due to change in hardware 
		     text spec by Ammon 10-3-94 */
                    y = buffer[i++];
                    for (x = 0; x < y; x += 2) {
                        str[x] = ((char *)buffer)[2 * i];
                        str[x + 1] = ((char *)buffer)[2 * i + 1];
                        i++;
                    }
                    str[y] = '\0';
                    fprintf(ofp,"\ngsave");
                    if (textangle != 0)
                        fprintf(ofp,"\n%d rotate", textangle);
                    fprintf(ofp, "\n32000 72 10 mul div 24000 72 10 mul div "
                            "scale");
                    fprintf(ofp, "\n(%s)show", str); 
                    fprintf(ofp, "\ngrestore");
                    break;
                case -6: /* Hardware text size */
                    i++;
                    /* (ignored) 
                       width = (float)buffer[i++] / 32000.0f; 
                       height = (float)buffer[i++] / 32000.0f;
                    */
                    i++;
                    i++;
                    break;
                case -7: /* Linestyle command */
                    i++;
                    EndPath(ofp, &fillFlag);
                    if (buffer[i] <= 0)
                        buffer[i] = 1;
                    fprintf(ofp, "\n%s",
                            linemodes[(buffer[i++] - 1) % NUMLSTYLS]);
                    break;
                case -8: /* Plot size command */
                    i++;
                    if (buffer[i] == 32000 && scale_flag != 0) {
                        printf("Scale size 32000 encountered.\n");
                        printf("Possible Old SGF - Scale anyway (y/n): ");
                        (void)fgets(scale, (int)sizeof(scale), stdin);
                        scale[strlen(scale) - 1] = '\0'; /* remove newline */
                    }
                    if (strncmp(scale, "n", 1) == 0 || scale_flag == 0)  {
                        i++; /* don't scale */
                    } else {
                        xsize = (float)buffer[i++] * 0.01f;
                        yscale = xscale = xsize / 10.0f;
                        fprintf(ofp, "\n%14.3f%14.3f scale", xscale, yscale);
                    }
                    break;
                case -9: /* Linewidth command */
                    i++;
                    EndPath(ofp, &fillFlag);
                    if (buffer[i] <= 0)
                        buffer[i] = 1;
                    new_width = buffer[i++];
                    if (new_width < 1 || new_width > 10)
                        new_width = 1;
                    new_width = new_width * base_width;
                    fprintf(ofp, "\n%d setlinewidth", new_width);
                    break;
                case -10: /* Polyfill Command */
                    i++;
                    igray = buffer[i++];
                    if (igray == 0)
                        gray_value = 1.0;
                    else if (igray == 1)
                        gray_value = 0.8;
                    else if (igray == 2)
                        gray_value = 0.0;
                    i++;
                    i++; /* Move command */
                    x = buffer[i++];
                    y = buffer[i++];
                    EndPath(ofp, &fillFlag);
                    fprintf(ofp, "\n%.1f g", gray_value);
                    fprintf(ofp, "\n%d %d m", x, y);
                    fillFlag = 1;
                    break;
                case -11: /* plot filled rectangle */
                    i++;
                    x = buffer[i++];
                    y = buffer[i++];
                    dx = buffer[i++];
                    dy = buffer[i++];
                    indx = buffer[i++];
                    if (indx < 0 || indx >= ict)
                    {
                        printf("Illegal color value: set to last index\n");
                        indx = ict - 1;
                    }
                    fprintf(ofp, "\n%d %d %d %d %g %g %g F", x, y, dx, dy,
                            red[indx], green[indx], blue[indx]);
                    break;
                case -12: /* set text angle */
                    i++;
                    textangle = buffer[i++];
                    break;
                case -13: /* color image */
                    iwidth = buffer[i++];
                    iheight = buffer[i++];
                    xloc = buffer[i++];
                    yloc = buffer[i++];
                    fprintf(ofp, "\n gsave");
                    fprintf(ofp, "\n /picstr %d string def", 3 * iwidth);
                    fprintf(ofp, "\n %d %d translate", xloc, yloc);
                    fprintf(ofp, "\n %d %d scale",32 * iwidth, 32 *iheight);
                    fprintf(ofp, "\n %d %d 8", iwidth, iheight);
                    fprintf(ofp, "\n [ %d 0 0 %d 0 %d ]", iwidth, -iheight,
                            iheight);
                    fprintf(ofp, "\n {currentfile\n picstr readhexstring pop}");
                    fprintf(ofp, "\n false 3");
                    fprintf(ofp, "\n colorimage\n\n");
                    imagecount = iwidth * iheight * 3;
                    doing_image = TRUE;
                    icount = 0;
                    break;
            case -14: /* RGB Fill */
              fprintf(ofp, "\nfill\n");
              break;
                default:
                    count = buffer[i++]; /* Skip unknown op-code */
                    printf("unknown op code %d, count = %d",
                           buffer[i - 2], count);
                    i += count;
            }
        }
    }
}


/* -------------------------------------------------------------------------- */

static void EndPath(FILE *ofp, int *fillFlag) {
    if (*fillFlag == 0)
        fprintf(ofp, "\ns");
    else if (*fillFlag == 1) {
        fprintf(ofp, "\nfill\n0.0 g");
        *fillFlag = 0;
    }
}

static int16_t int_join16(char cbuf[]) {
    union {
        char cval[2];
        int16_t lval;
    } l_union;
    
    l_union.cval[0] = cbuf[0];
    l_union.cval[1] = cbuf[1];
    return(l_union.lval);
}

static int16_t int_swap16(char cbuf[]) {
    union {
        char cval[2];
        int16_t lval;
    } l_union;
    
    l_union.cval[1] = cbuf[0];
    l_union.cval[0] = cbuf[1];
    return(l_union.lval);
}

static int32_t int_swap32(char cbuf[]) {
    union {
        char cval[4];
        unsigned int lval;
    } l_union;
    
    l_union.cval[3] = cbuf[0];
    l_union.cval[2] = cbuf[1];
    l_union.cval[1] = cbuf[2];
    l_union.cval[0] = cbuf[3];
    return(l_union.lval);
}

static int32_t int_join32(char cbuf[]) {
    union {
        char cval[4];
        unsigned int lval;
    } l_union;
    
    l_union.cval[3] = cbuf[3];
    l_union.cval[2] = cbuf[2];
    l_union.cval[1] = cbuf[1];
    l_union.cval[0] = cbuf[0];
    return(l_union.lval);
}

/* eof */
