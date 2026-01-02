#include <Python.h>

static struct PyModuleDef c_ref_backend_module = {
    PyModuleDef_HEAD_INIT,
    "_c_ref_backend",
    "Reference backend C extension",
    -1,
    NULL,
};

PyMODINIT_FUNC PyInit__c_ref_backend(void) {
    return PyModule_Create(&c_ref_backend_module);
}
