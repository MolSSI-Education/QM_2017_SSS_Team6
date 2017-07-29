//Header files
#include <iterator>
#include <fstream>
#include <iostream>
#include <lawrap/blas.h>
#include <vector>

typedef std::vector<double> dvector;

//Function initialization
dvector get_data(std::string file_name);
void print_as_matrix(dvector x);

//Main Function
int main() {

dvector S, C, H, F;
S = get_data("S.data");
C = get_data("C.data");
H = get_data("H.data");
F = get_data("F.data");

int norb = sqrt(C.size());
int nocc = 5;
dvector D(C.size());
LAWrap::gemm('T','N',norb , norb, nocc, 2.0, C.data(),norb, C.data(),norb, 0.0, D.data(), norb);

std::cout<<"D:"<<std::endl;
print_as_matrix(D);

std::cout<<"\nH:"<<std::endl;
print_as_matrix(H);

// The H matrix is overwritten here: H = F + H
LAWrap::axpy(C.size(), 1.0, F.data(), 1, H.data(), 1);

std::cout<<"\n F + H:"<<std::endl;
print_as_matrix(H);

dvector FHD(C.size());
LAWrap::gemm('N','N', norb, norb, norb, 0.5, H.data(), norb, D.data(), norb, 0.0, FHD.data(), norb);

  auto sum = 0.0;
  for (size_t i = 0; i <norb; i++) {
    sum += FHD[i+i*norb];
  }

  std::cout<<"\nElectronic energy: "<<sum;
  sum += 8.00236645071908;
  std::cout<<"\tTotal energy: "<<sum<<std::endl;

  return 0;
}

//Function declarations
dvector get_data(std::string file_name){
  dvector vec;
  std::ifstream input_file(file_name);
  std::copy(
      std::istream_iterator<double>(input_file),
      std::istream_iterator<double>(),
      std::back_inserter(vec));
return vec;
}

void print_as_matrix(dvector x){
  int vector_length = std::sqrt(x.size());
  for (int i = 0; i < x.size(); i++) {
    printf("%.8 f\t", x[i]);
    if ((i+1) % vector_length == 0 ) {
      std::cout << std::endl;
    }
  }
}
