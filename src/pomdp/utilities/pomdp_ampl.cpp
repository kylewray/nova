/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2017 Kyle Hollins Wray, University of Massachusetts
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *  the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 *  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 *  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 *  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#include <nova/pomdp/utilities/pomdp_ampl.h>

#include <nova/error_codes.h>
#include <nova/constants.h>

#include <cstdio>
#include <fstream>
#include <string>

namespace nova {

int pomdp_ampl_save_data_file(const POMDP *pomdp, unsigned int k, const char *path)
{
    std::string filename(path);
    filename += "/nova_nlp_ampl.dat";

    std::ofstream file(filename, std::ofstream::out);
    if (!file.is_open()) {
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    file << "data;" << std::endl;

    file << "set STATES :=";
    for (unsigned int s = 0; s < pomdp->n; s++) {
        file << " s" << s;
    }
    file << ";" << std::endl;

    file << "set ACTIONS :=";
    for (unsigned int a = 0; a < pomdp->m; a++) {
        file << " a" << a;
    }
    file << ";" << std::endl;

    file << "set OBSERVATIONS :=";
    for (unsigned int o = 0; o < pomdp->z; o++) {
        file << " o" << o;
    }
    file << ";" << std::endl;

    file << "set CONTROLLER_NODES :=";
    for (unsigned int i = 0; i < k; i++) {
        file << " q" << i;
    }
    file << ";" << std::endl;

    for (unsigned int q = 0; q < k; q++) {
        file << "let q0[\"q" << q << "\"] := ";
        if (q == 0) {
            file << "1.0";
        } else {
            file << "0.0";
        }
        file << ";" << std::endl;
    }

    for (unsigned int s = 0; s < pomdp->n; s++) {
        file << "let b0[\"s" << s << "\"] := " << pomdp->B[0 * pomdp->n + s] << ";" << std::endl;
    }

    file << "let gamma := " << pomdp->gamma << ";" << std::endl;

    for (unsigned int s = 0; s < pomdp->n; s++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {
            for (unsigned int i = 0; i < pomdp->ns; i++) {
                int sp = pomdp->S[s * pomdp->m * pomdp->ns + a * pomdp->ns + i];
                if (sp < 0) {
                    break;
                }

                file << "let T[\"s" << s << "\", \"a" << a << "\", \"s" << sp << "\"] := ";
                file << pomdp->T[s * pomdp->m * pomdp->ns + a * pomdp->ns + i] << ";" << std::endl;
            }
        }
    }

    for (unsigned int a = 0; a < pomdp->m; a++) {
        for (unsigned int sp = 0; sp < pomdp->n; sp++) {
            for (unsigned int o = 0; o < pomdp->z; o++) {
                file << "let O[\"a" << a << "\", \"s" << sp << "\", \"o" << o << "\"] := ";
                file << pomdp->O[a * pomdp->n * pomdp->z + sp * pomdp->z + o] << ";" << std::endl;
            }
        }
    }

    for (unsigned int s = 0; s < pomdp->n; s++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {

            file << "let R[\"s" << s << "\", \"a" << a << "\"] := ";
            file << pomdp->R[s * pomdp->m + a] << ";" << std::endl;
        }
    }

    file.close();

    return NOVA_SUCCESS;
}

}; // namespace nova

