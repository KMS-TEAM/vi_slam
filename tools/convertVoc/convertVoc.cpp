//
// Created by cit-industry on 28/06/2021.
//

#include <time.h>

#include "../thirdparty/DBoW3/DBoW3/src/DBoW3.h"

using namespace std;

bool load_as_text(DBoW3::Vocabulary* voc, const std::string infile) {
    clock_t tStart = clock();
    bool res = voc->loadFromTextFile(infile);
    printf("Loading fom text: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    return res;
}

void load_as_xml(DBoW3::Vocabulary* voc, const std::string infile) {
    clock_t tStart = clock();
    voc->load(infile);
    printf("Loading fom xml: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

void load_as_binary(DBoW3::Vocabulary* voc, const std::string infile) {
    clock_t tStart = clock();
    voc->load(infile);
    printf("Loading fom binary: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

void save_as_xml(DBoW3::Vocabulary* voc, const std::string outfile) {
    clock_t tStart = clock();
    voc->save(outfile);
    printf("Saving as xml: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

void save_as_text(DBoW3::Vocabulary* voc, const std::string outfile) {
    clock_t tStart = clock();
    voc->saveToTextFile(outfile);
    printf("Saving as text: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

void save_as_binary(DBoW3::Vocabulary* voc, const std::string outfile) {
    clock_t tStart = clock();
    voc->save(outfile, true);
    printf("Saving as binary: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}


int main(int argc, char **argv) {
    cout << "BoW load/save benchmark" << endl;
    DBoW3::Vocabulary* voc = new DBoW3::Vocabulary();

    load_as_text(voc, "Vocabulary/ORBvoc.txt");
    save_as_binary(voc, "Vocabulary/ORBvoc.bin");

    return 0;
}


