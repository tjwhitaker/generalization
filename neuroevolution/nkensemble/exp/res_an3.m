% Compute the average results for Committee of ANNs

clear; close all; clc;

n_runs=100;
%K=[2:8]';
K=[2:8]';
reservoir=1;
regular=1;  %regular (1) or random (0)

if (regular==1)
    file_s2=fopen(strcat('nnetRes_',num2str(reservoir),'_reg.txt'),'w');
else
    file_s2=fopen(strcat('nnetRes_',num2str(reservoir),'_ran.txt'),'w');
end
N=100;
cd (strcat('reservoir_',num2str(reservoir)))

for ki=1:size(K,1),
    cd (strcat('k',num2str(K(ki))));
    if (regular==1)
        cd regular;
    else
        cd random;
    end

    n_subf=2^(K(ki)+1);
        
    load best_out.dat;
    load best_com.dat;
    load best_co1.dat;
           
    for run=0:n_runs-1,
        
        % Individual contributions
        file_W=fopen(strcat('NK_',num2str(run),'.dat'),'r');
        As=fscanf(file_W,'%s/n');
        while(strcmp(As,'F_el')==0)
            As=fscanf(file_W,'%s/n');
        end
        for i=1:N
            for j=1:n_subf,
                Fel(i,j)=fscanf(file_W,'%f/n');
            end
        end
        fclose(file_W);
        max_Fel(run+1)=max(max(Fel));
        %disp(strcat('Adjacent: K=',num2str(K(ki)),' Run=',num2str(run),' max(Fel)=', num2str(max_Fel(run+1)),' Committee=',  num2str(best_com(run+1))));
     
    end
    if (regular==1)
        data_file2 = strcat( 'regular & ',num2str(reservoir), ' & ',num2str(K(ki)), ' & ', num2str(mean(max_Fel),'%.2f'),' $\pm$ ',num2str(std(max_Fel),'%.2f'),' & ' ,...
                                                num2str(mean(best_out),'%.0f'),' $\pm$ ',num2str(std(best_out),'%.0f'),' & ' ,num2str(max(best_out)),' & ' ,...
                                                num2str(mean(best_com),'%.0f'),' $\pm$ ',num2str(std(best_com),'%.0f'),' & ' ,num2str(max(best_com)),' & ' ,...
                                                num2str(mean(best_co1),'%.0f'),' $\pm$ ',num2str(std(best_co1),'%.0f'));
    else                                      
        data_file2 = strcat( 'random & ',num2str(reservoir), ' & ',num2str(K(ki)), ' & ', num2str(mean(max_Fel),'%.2f'),' $\pm$ ',num2str(std(max_Fel),'%.2f'),' & ' ,...
                                                num2str(mean(best_out),'%.2f'),' $\pm$ ',num2str(std(best_out),'%.2f'),' & ' ,num2str(max(best_out)),' & ' ,...
                                                 num2str(mean(best_com),'%.2f'),' $\pm$ ',num2str(std(best_com),'%.2f'),' & ' ,num2str(max(best_com)),' & ' ,...
                                                num2str(mean(best_co1),'%.2f'),' $\pm$ ',num2str(std(best_co1),'%.2f'));
    end
    
  
        [p1,h1] = signrank(best_co1,best_out,0.05)
        [p2,h2] = signrank(best_co1,best_com,0.05)
        if (h1==1 && h2==1) 
            data_file2 = strcat( data_file2,' (s) ');
        end
 
    data_file2 = strcat( data_file2,' & ' ,num2str(max(best_co1)),' \\');
    fprintf(file_s2, '%s \n',data_file2);
    
    %disp(strcat('Adjacent: K=',num2str(K(ki)),', Mean:  max(Fel)=', num2str(mean(max_Fel)),' Committee=',  num2str(mean(best_com))));

    cd ..    
    cd ..
end
cd ..

fclose(file_s2);
        
