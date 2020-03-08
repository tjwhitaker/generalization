% Compute the average results for Committee of ANNs

clear; close all; clc;

n_runs=100;
%K=[2:8]';
K=[3]';
reservoir=1;
regular=1;  %regular (1) or random (0)
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
        
    load res_test.dat;

    k=1;
    for i=1:625,
        for j=1:100,
            x(i,j)=res_test(k);
            k=k+1;
        end 
    end
    y=zeros(625,1);
    cont=0;
    for j=1:100,
        %sum(x(:,j))
        if (sum(x(:,j))==525)
            y=y+(1-x(:,j));
            cont=cont+1;
        end
    end
    t=1:625;
    figure (1),
        set(gcf,'DefaultLineLineWidth',1.5);
        set(gcf,'DefaultLineMarkerSize',10.0);
        axes('FontSize',20);
        subplot(5,1,1)
            plot(t,y,'b-');
            ylabel('fails','FontSize',20 )
             axis([0 626 -1 cont+1])
   
    cont=1;
    for i0=1:5,
        for i1=1:5,
          for i2=1:5,
              for i3=1:5,
                  z(1,cont)=(i0-3)/2;
                  z(2,cont)=(i1-3)/2;
                  z(3,cont)=(i2-3)/2;
                  z(4,cont)=(i3-3)/2;
                  cont=cont+1;
              end
          end
        end
    end
    

        subplot(5,1,2)
            plot(t,z(1,:),'g-');
             ylabel('cart pos.','FontSize',20 )
             axis([0 626 -1.1 1.1])
        subplot(5,1,3)
            plot(t,z(2,:),'g-');
            ylabel('cart vel.','FontSize',20 )
            axis([0 626 -1.1 1.1])
        subplot(5,1,4)
            plot(t,z(3,:),'g-');
            ylabel('pole 1 pos.','FontSize',20 )
            axis([0 626 -1.1 1.1])
        subplot(5,1,5)
            plot(t,z(4,:),'g-');
            ylabel('pole 1 vel.','FontSize',20 )
           axis([0 626 -1.1 1.1])
        xlabel('initial configuration','FontSize',20 )
    
    cd ..    
    cd ..
end
cd ..


        
