#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>
#include <vector>

namespace py = pybind11;
std::vector<py::array> form_JK(py::array_t<double> I,
                                       py::array_t<double> D)
{
    py::buffer_info D_info = D.request();
    py::buffer_info I_info = I.request();
    size_t n = D_info.shape[0];  

    const double*D_data = static_cast<double*>(D_info.ptr);
    const double*I_data = static_cast<double*>(I_info.ptr);

    std::vector<double> J_data(n*n);
    std::vector<double> K_data(n*n);
    size_t n3 = n*n*n;
    size_t n2 = n*n;
   // Form J and K


    for(size_t p = 0; p < n; p++)
    {
        size_t pn3 = p*n3;
	size_t pn = p * n;
        for(size_t q = p; q < n; q++)
        {
            size_t pn3qn2 = pn3 + q * n2;
            for(size_t r = 0; r < n; r++)
            {
                size_t pn3qn2rn = pn3qn2 + r * n;
                for(size_t s = r; s < n; s++)
                {
                    //J_data[p*n + q] += I_data[p * n3 + q * n2 + r * n + s] * D_data[r * n + s]*2.0;
                    J_data[pn + q] += I_data[pn3qn2rn + s] * D_data[r * n + s]*2.0;
                }
            }
        J_data[q*n + p] = J_data[p*n + q];
        }
    
        for(size_t q = p; q < n; q++)
        {
            size_t pn3qn = pn3 + q * n;
            for(size_t r = 0; r < n; r++)
            {
                size_t pn3qnrn2 = pn3qn + r * n2;
                for(size_t s = 0; s < n; s++)
                {
                    if(s == r)
                    {
                    //K_data[p*n + q] += I_data[p * n3 + r * n2 + q * n + s] * D_data[r * n + s]*2.0;
                    K_data[pn + q] += I_data[pn3qnrn2 + s] * D_data[r * n + s]*2.0;
                    }
                    else
                    {
                    //K_data[p*n + q] += I_data[p * n3 + r * n2 + q * n + s] * D_data[r * n + s];
                    K_data[pn + q] += I_data[pn3qnrn2 + s] * D_data[r * n + s];
                    }
                    
                }
            }
            K_data[q*n + p] = K_data[pn + q];
        }
    }
     
    py::buffer_info Jbuf =
        {
        J_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {n,n},
        {n * sizeof(double), sizeof(double)}
        };

    py::buffer_info Kbuf =
        {
        K_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {n,n},
        {n * sizeof(double), sizeof(double)}
        };

    py::array J(Jbuf);
    py::array K(Kbuf);
    return {J, K};
}


PYBIND11_PLUGIN(jk)
{
    py::module m("jk", "hi");
    m.def("form_JK", &form_JK, "this is J");
    return m.ptr();
}
