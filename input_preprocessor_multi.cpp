#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
#include <cmath>
using namespace std;

int main(int argc, char * argv[]){
    float start_time = 0;
    float end_time = 0;
    char prefix[] = {'a','b','c','d','e','f','g','i','l','m','n','o','p','q','r','s','t','u','v','w','x','y'};
    omp_set_num_threads(22);
    #pragma omp parallel for
    for(int i = 0; i < sizeof(prefix); i++){ // i = arquivo
        string points[1000][20][3]; // 1000 linhas x 20 pontos x 3 coordenadas
        string new_line = "";
        int gesture = i;
        ifstream input_file;
        string line;
        std::vector<string>  line_array;
        string input_filename = "raw/raw_";
        input_filename.append(1,prefix[i]);
        input_filename.append(".txt");
        input_file.open(input_filename);
        ofstream output_file;
        string output_filename = "normalized_test/";
        output_filename.append(1,prefix[i]);
        output_filename.append(".txt");
        output_file.open(output_filename);
        //while(getline(input_file,line)){ // para cada linha no arquivo
        for(int j = 0; j < 1000; j++){
            getline(input_file,line);
            //std::cout << j << std::endl;
            string coord;
            stringstream ss(line);
            while (getline(ss, coord, ',')){
                line_array.push_back(coord);
            }
            int p = 0; // p = ponto
            for(int k = 0; k < 60; k += 3){ // para cada ponto na linha
                points[j][p][0] = line_array[k];
                points[j][p][1] = line_array[k+1];
                points[j][p][2] = line_array[k+2];
                p += 1;
            }
            line_array = {};
            //std::cout << points[j][0][1] << std::endl;
            float a,b;
            a = std::stof(points[j][8][0]);
            b = std::stof(points[j][8][1]);
            float c2 = a*a + b*b;
            float c = sqrt(c2);
            float alpha = acos(b/c); // * (180.0 / M_PI); // 0 -> 90 -> 180/180 -> 90 -> 0
            float angle;
            if(a > 0)
                angle = M_PI - alpha;
            else
                angle = alpha;
            float ref_angle = M_PI/2 - angle;
            float ref_scale = c / 200;
            for(int l = 0; l < 20; l++){
                float x,y;
                x = std::stof(points[j][l][0]) / ref_scale;
                y = std::stof(points[j][l][1]) / ref_scale;
                float x2 = (x*cos(ref_angle)) - (y*sin(ref_angle));
                float y2 = (x*sin(ref_angle)) + (y*cos(ref_angle));
                new_line += std::to_string(x2)+","+std::to_string(y2)+","+points[j][l][2]+",";
            }
            new_line += std::to_string(gesture)+"\n";
            output_file << new_line;
            new_line = "";
        }
        std::cout << i << std::endl;
        input_file.close();
        output_file.close();
    }
    return 0;
}
