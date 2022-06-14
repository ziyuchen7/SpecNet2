close all;
mat1 = load('./disc.mat');
mat2 = load('../test.mat');
mat3 = load('./specnet1_localu_unit_256_depth_3_lr_0.0001_batch_45-10.mat');
mat4 = load('./specnet2_neighbor_unit_256_depth_3_lr_0.0001_batch_2-20.mat');

labelX = mat1.label;
labelX2 = mat2.label;

fontsize = 24;


evecs = mat1.evecs(:,2:7);
    
Psi3 = mat3.y(:,2:7)/sqrt(20000);
Psi_sub3 = mat3.testy(:,2:7)/sqrt(20000);
Psi4 = mat4.y/sqrt(20000);
Psi_sub4 = mat4.testy/sqrt(20000);

c3 = sign(diag(evecs'*Psi3));
c4 = sign(diag(evecs'*Psi4));


figure()
scatter3(evecs(:,1),evecs(:,2),evecs(:,3),40,labelX,'o');
grid on; colormap(jet); colorbar();
title('2nd-4th eigenvectors of $D^{-1}W$','Interpreter','latex');
set(gca, 'fontsize', fontsize);
view(30,30)

figure()
scatter3(evecs(:,4),evecs(:,5),evecs(:,6),40,labelX,'o');
grid on; colormap(jet); colorbar();
title('5th-7th eigenvectors of $D^{-1}W$','Interpreter','latex');
set(gca, 'fontsize', fontsize);
view(30,30)

figure()
scatter3(c3(1)*Psi3(:,1),c3(2)*Psi3(:,2),c3(3)*Psi3(:,3),40,labelX,'o');
grid on; colormap(jet); colorbar();
title({'SpecNet1-local (training)', '1st-3rd coordinates'},'Interpreter','latex');
set(gca, 'fontsize', fontsize);
view(30,30)

figure()
scatter3(c3(4)*Psi3(:,4),c3(5)*Psi3(:,5),c3(6)*Psi3(:,6),40,labelX,'o');
grid on; colormap(jet); colorbar();
title({'SpecNet1-local (training)','4th-6th coordinates'},'Interpreter','latex');
set(gca, 'fontsize', fontsize);
view(30,30)

figure()
scatter3(c3(1)*Psi_sub3(:,1),c3(2)*Psi_sub3(:,2),c3(3)*Psi_sub3(:,3),40,labelX2,'o');
grid on; colormap(jet); colorbar();
title({'SpecNet1-local (testing)', '1st-3rd coordinates'},'Interpreter','latex');
set(gca, 'fontsize', fontsize);
view(30,30)

figure()
scatter3(c3(4)*Psi_sub3(:,4),c3(5)*Psi_sub3(:,5),c3(6)*Psi_sub3(:,6),40,labelX2,'o');
grid on; colormap(jet); colorbar();
title({'SpecNet1-local (testing)', '4th-6th coordinates'},'Interpreter','latex');
set(gca, 'fontsize', fontsize);
view(30,30)

figure()
scatter3(c4(1)*Psi4(:,1),c4(2)*Psi4(:,2),c4(3)*Psi4(:,3),40,labelX,'o');
grid on; colormap(jet); colorbar();
title({'SpecNet2-neighbor (training)', '1st-3rd coordinates'},'Interpreter','latex');
set(gca, 'fontsize', fontsize);
view(30,30)

figure()
scatter3(c4(4)*Psi4(:,4),c4(5)*Psi4(:,5),c4(6)*Psi4(:,6),40,labelX,'o');
grid on; colormap(jet); colorbar();
title({'SpecNet2-neighbor (training)', '4th-6th coordinates'},'Interpreter','latex');
set(gca, 'fontsize', fontsize);
view(30,30)

figure()
scatter3(c4(1)*Psi_sub4(:,1),c4(2)*Psi_sub4(:,2),c4(3)*Psi_sub4(:,3),40,labelX2,'o');
grid on; colormap(jet); colorbar();
title({'SpecNet2-neighbor (testing)', '1st-3rd coordinates'},'Interpreter','latex');
set(gca, 'fontsize', fontsize);
view(30,30)

figure()
scatter3(c4(4)*Psi_sub4(:,4),c4(5)*Psi_sub4(:,5),c4(6)*Psi_sub4(:,6),40,labelX2,'o');
grid on; colormap(jet); colorbar();
title({'SpecNet2-neighbor (testing)', '4th-6th coordinates'},'Interpreter','latex');
set(gca, 'fontsize', fontsize);
view(30,30)



