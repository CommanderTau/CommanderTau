#include <iostream>
#include <ofstream>

int main()
{
    std::ofstream out;          // поток для записи
    out.open("hello.txt");      // открываем файл для записи
    char c;
    char l;
    std::string s = "string";
    l = out.size();
    for(int i=0; i<l-1; i++) {
        c[i] = s[i];
    }
    out.close(); 
    std::cout << "File has been written" << std::endl;
}
