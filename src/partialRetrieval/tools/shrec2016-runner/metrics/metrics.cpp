// Original source: Pratikakis et al.
// Used for computing several metrics for the SHREC2016 partial object retrieval track
// I have cleaned up the source code a bit, and added two convenience command line parameters.
// Other than formatting, adding compatibility with linux, and renaming a few variables, 
// the program's functionality is completely unaltered and should produce bit-for-bit equivalent results
// compared to the version that comes with the benchmark evaluation archive. 

//#include <windows.h>
#include <math.h>
//#include <process.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>


using namespace std;
#define PI 3.14159265

#include <time.h>
#ifndef M_LOG2E
#define M_LOG2E 1.44269504088896340736 //log2(e)
#endif
inline double log2(const double x){
    return  log(x) * M_LOG2E;
}

int main(int argc,char* argv[]) {

    printf("Metrics for Hampson museum database!\n");

    if (argc != 6) {
        printf("Wrong syntax.\nCorrect syntax is ./metrics queries_cla.txt targets_cla.txt ranked_lists_directory queryCount output_file_name.txt\n");
        return (-1);
    } else if (argc == 6) //main functionality
    {

        const int num_of_queries = std::stoi(std::string(argv[4]));//set number of queries, 21 for artificial, 25 for real low and high quality
        const int num_of_targets = 383;//set number of targets


        ifstream qfile;
        qfile.open(argv[1]);
        if (!qfile.is_open()) {
            std::cout << "Failed to open query labels file!" << std::endl;
            return 0;
        }

        ifstream tfile;
        tfile.open(argv[2]);
        if (!tfile.is_open()) {
            std::cout << "Failed to open target labels file!" << std::endl;
            return 0;
        }

        int queries[num_of_queries][2];
        int targets[num_of_targets][2];

        string name_string;
        //std::cout << "Query file contents: " << std::endl;
        for (int i = 0; i < num_of_queries; i++) {
            qfile >> name_string >> queries[i][1] >> queries[i][0];
            //std::cout<<queries[i][1]<<" "<<queries[i][0]<<"\n";

        }
        //std::cout << std::endl;

        //std::cout << "Target file contents: " << std::endl;
        for (int i = 0; i < num_of_targets; i++) {
            tfile >> name_string >> targets[i][1] >> targets[i][0];
            //std::cout<<targets[i][1]<<" "<<targets[i][0]<<"\n";

        }
        //std::cout << std::endl;

        qfile.close();
        tfile.close();


        string ranked_lists_dir = argv[3];

        string rank_name;

        string rquery, rtarget, rtarget_sub;
        double distance;


        //variables and output files for storing metrics results
        int NN = 0;
        double FT[num_of_queries] = {0.0};
        double ST[num_of_queries] = {0.0};
        double FT_Average = 0.0;
        double ST_Average = 0.0;
        int class_cardinality[24] = {0};
        double Precision[num_of_queries][num_of_targets] = {1.0};
        double Recall[num_of_queries][num_of_targets] = {0.0};
        double P_Interpolated[num_of_queries][num_of_targets] = {1.0};
        double R_Interpolated[num_of_queries][num_of_targets] = {1.0};
        double P_Average[num_of_queries] = {0.0};
        double R_Average[num_of_queries] = {0.0};
        double P32, R32, EM;
        double DCG[num_of_queries] = {1.0};
        double DCGmax[num_of_queries] = {1.0};
        double DCGaverage = 0.0;

        //ofstream nnfile;
        //nnfile.open("metrics_results.txt");
        //if(!nnfile.is_open()){return 0;}


        //variables and output files for storing metrics results


        for (int i = 0; i < num_of_queries; i++)    //code for NN, calculating cardinalities

        {
            rank_name.clear();
            rank_name = ranked_lists_dir;
            if (i + 1 < 10)
                rank_name += "/P0000";
            else
                rank_name += "/P000";

            string num;
            num = std::to_string(i + 1);

            rank_name += num;
            //cout<<rank_name<<"\n";

            ifstream rfile;
            rfile.open(rank_name);
            if (!rfile.is_open()) {
                std::cout << "Failed to open ranked lists file: " << rank_name << std::endl;
                return 0;
            }


            rfile >> rquery;

            for (int j = 0; j < num_of_targets; j++) {
                rfile >> rtarget >> distance;
                rtarget_sub = rtarget.substr(3, 3);


                if (j == 0)//code for NN
                {
                    //cout << queries[i][1]
                    //     << "  " << atoi(rtarget_sub.c_str())
                    //     << " " << targets[atoi(rtarget_sub.c_str()) - 1][1] << "\n";
                    if (queries[i][1] == targets[atoi(rtarget_sub.c_str()) - 1][1])
                        NN++;
                }//code for NN

                if (i ==
                    0)//code for counting cardinality of each class. Counting targets from first query (i==0) is sufficient (for i>0 targets are just the same)
                    class_cardinality[targets[atoi(rtarget_sub.c_str()) - 1][1]]++;

            }
            rfile.close();

        }//end of for all queries



        //code for NN, calculating cardinalities

        ofstream prfile;
        prfile.open(argv[5]);
        if (!prfile.is_open()) {
            std::cout << "Failed to write precision-recall metrics file" << std::endl;
            return 0;
        }

        for (int queryIndex = 0; queryIndex < num_of_queries; queryIndex++) {
            rank_name.clear();
            rank_name = ranked_lists_dir;
            if (queryIndex + 1 < 10)
                rank_name += "/P0000";
            else
                rank_name += "/P000";

            string num;
            num = std::to_string(queryIndex + 1);

            rank_name += num;
            //cout<<rank_name<<"\n";

            ifstream rfile;
            rfile.open(rank_name);
            if (!rfile.is_open()) { return 0; }

            rfile >> rquery;

            int interpolation_counter = 0;
            for (int resultIndex = 0; resultIndex < num_of_targets; resultIndex++) {
                rfile >> rtarget >> distance;
                rtarget_sub = rtarget.substr(3, 3);

                //if (i==0)
                //	std::cout<<"A:"<<Precision[i][resultIndex]<<"\n";

                if (resultIndex == 0) {
                    if (queries[queryIndex][1] == targets[atoi(rtarget_sub.c_str()) - 1][1]) {
                        Precision[queryIndex][resultIndex] = 1.0;
                        Recall[queryIndex][resultIndex] = 1.0 / class_cardinality[queries[queryIndex][1]];
                        P_Interpolated[queryIndex][interpolation_counter] = Precision[queryIndex][resultIndex];
                        R_Interpolated[queryIndex][interpolation_counter] = Recall[queryIndex][resultIndex];
                    } else {
                        Precision[queryIndex][resultIndex] = 0.0;
                        Recall[queryIndex][resultIndex] = 0.0;
                        P_Interpolated[queryIndex][interpolation_counter] = Precision[queryIndex][resultIndex];
                        R_Interpolated[queryIndex][interpolation_counter] = Recall[queryIndex][resultIndex];
                    }
                } else if (queries[queryIndex][1] == targets[atoi(rtarget_sub.c_str()) - 1][1]) {
                    Precision[queryIndex][resultIndex] =
                            Precision[queryIndex][resultIndex - 1] * (double) (resultIndex) /
                            (double) (resultIndex + 1) + 1.0 / ((double) (resultIndex + 1));//
                    Recall[queryIndex][resultIndex] =
                            Recall[queryIndex][resultIndex - 1] + 1.0 / class_cardinality[queries[queryIndex][1]];//
                    interpolation_counter++;
                    P_Interpolated[queryIndex][interpolation_counter] = Precision[queryIndex][resultIndex];
                    R_Interpolated[queryIndex][interpolation_counter] = Recall[queryIndex][resultIndex];
                } else {
                    Precision[queryIndex][resultIndex] =
                            Precision[queryIndex][resultIndex - 1] * (double) (resultIndex) /
                            (double) (resultIndex + 1);//
                    Recall[queryIndex][resultIndex] = Recall[queryIndex][resultIndex - 1];
                }

                if (resultIndex < class_cardinality[queries[queryIndex][1]])//code for FT
                    if (queries[queryIndex][1] == targets[atoi(rtarget_sub.c_str()) - 1][1])
                        FT[queryIndex]++;

                if (resultIndex < 2 * class_cardinality[queries[queryIndex][1]])//code for ST
                    if (queries[queryIndex][1] == targets[atoi(rtarget_sub.c_str()) - 1][1])
                        ST[queryIndex]++;

                if (resultIndex > 0)//code for DCG
                {
                    DCG[queryIndex] = DCG[queryIndex] +
                                      (double) (queries[queryIndex][1] == targets[atoi(rtarget_sub.c_str()) - 1][1]) /
                                      log2((double) (resultIndex + 1)); //(log10(i)/log10(2));
                }

                if ((resultIndex < class_cardinality[queries[queryIndex][1]]) && (resultIndex > 0)) {
                    DCGmax[queryIndex] = DCGmax[queryIndex] + 1.0 / log2((double) (resultIndex + 1));
                }    //code for DCG

            }
            //cout<<"DCGi:"<<DCG[i]<<"DCGmax[i]:"<<DCGmax[i]<<"\n";

            //int k;//Write PR interpolated values for first (i=0) query
            //if (i==0)
            //{
            //	for (k=0;k<=num_of_targets;k++)
            //		//prfile<<Recall[i][k]<<" "<<Precision[i][k]<<"\n";
            //		prfile<<R_Interpolated[i][k]<<" "<<P_Interpolated[i][k]<<"\n";
            //
            //	if (k!=num_of_targets)
            //		prfile<<R_Interpolated[i][num_of_targets]<<" "<<P_Interpolated[i][num_of_targets]<<"\n";

            //}//Write PR interpolated values for first (i=0) query

            rfile.close();
            FT[queryIndex] = FT[queryIndex] / class_cardinality[queries[queryIndex][1]];
            ST[queryIndex] = ST[queryIndex] / class_cardinality[queries[queryIndex][1]];

            FT_Average += FT[queryIndex] / (double) (num_of_queries);
            ST_Average += ST[queryIndex] / (double) (num_of_queries);

            DCGaverage += (DCG[queryIndex] / DCGmax[queryIndex]) / (double) (num_of_queries);

        }

        //Calculate and write average PR value
        int stop_search;
        for (int k = 1; k <= 20; k++) {
            R_Average[k] = double(k) * 0.05;
            for (int i = 0; i < num_of_queries; i++) {
                int j = 0;
                stop_search = 0;
                while (!stop_search)
                    if (R_Interpolated[i][j] >= double(k - 1) * 0.05) {
                        P_Average[k] += P_Interpolated[i][j] / (double) (num_of_queries);
                        stop_search = 1;
                    } else
                        j++;
            }
        }

        prfile << "Average Precision-recall:" << std::endl;

        for (int k = 2; k <= 20; k = k + 2)//Write
            //prfile<<R_Average[k]<<" "<<P_Average[k]<<"\n";
            prfile << P_Average[k] << "\n";

        //Calculate and write average PR value

        prfile << std::endl << std::endl << "Computed metrics:" << std::endl;
        //code for P-R

        P32 = P_Average[(int) ((32.0 / (double) (num_of_targets)) * 20.0)];
        R32 = R_Average[(int) ((32.0 / (double) (num_of_targets)) * 20.0)];
        EM = 2.0 / (1.0 / P32 + 1.0 / R32);
        
        std::cout << "Nearest Neighbors:" << NN << " " << ((double) NN / (double) (num_of_queries)) * 100.0 << "% \n";
        std::cout << "First Tier:" << FT_Average << " " << " \n";
        std::cout << "Second Tier:" << ST_Average << " " << " \n";
        std::cout << "E-Measure:" << EM << "\n";
        std::cout << "DCG:" << DCGaverage << "\n";

        prfile << "Nearest Neighbors:" << NN << " " << ((double) NN / (double) (num_of_queries)) * 100.0 << "% \n";
        prfile << "First Tier:" << FT_Average << " " << " \n";
        prfile << "Second Tier:" << ST_Average << " " << " \n";
        prfile << "E-Measure:" << EM << "\n";
        prfile << "DCG:" << DCGaverage << "\n";

        prfile.close();
        //nnfile.close();
    }//end else-main functionality



    printf("End of Computation\n");

    return 0;
}


