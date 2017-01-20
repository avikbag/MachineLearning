%{
%Simple Script to Parse a Data File in a particular format and display it's
content in a 2D point graph

%Matt Burlick, mjburlick@drexel.edu
%5/20/2016

%}

clear all; %remove all the old variables in the workspace
close all;

filename = 'toy.data';  %the file to read
datafile = 'toy.mat';
if(exist(datafile,'file'))
    load(datafile);
else    
    fid = fopen(filename);  %open it
    if(fid<0)  %make sure it exists
        display('File not found');
        return;
    end

    fgetl(fid);%remove the first line
    x=[];  %x will be an Nx2 matrix storing the features
    y=[];  %y will be an Nx1 matrix storing the classes

    while(~feof(fid))  %while we're not at the end-of-file
        line = fgetl(fid);  %read the next line
        C = textscan(line,'"%d" %f %f "%d"');  %parse it according to the format
        x(end+1, :) = [C{2}, C{3}];  %add a row to x and put the content of the 2nd and 3rd cells in it
        y(end+1,1) = C{4};    %add a row to y and add the content of the 4th cell in it
    end
    fclose(fid);  %don't forget to close your file!
    save(datafile,'x','y');
end %end of if(exist.... else...
figure(1);  %plot in figure 1
plot(x(y==-1,1), x(y==-1,2),'or');  %plot as red circles the first feature vs the second one for instances who's y value is -1
hold on;                            %draw the next thing on top of this one
plot(x(y==1,1), x(y==1,2),'xb');  %plot as red circles the first feature vs the second one for instances who's y value is 1
hold off;                   %future plots will erase the content of this one
%add some nice labels
title('My First Graph');
xlabel('x_1');
ylabel('x_2');
legend('-1', '+1');

