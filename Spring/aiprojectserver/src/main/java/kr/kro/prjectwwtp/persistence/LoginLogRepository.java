package kr.kro.prjectwwtp.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.LoginLog;

public interface LoginLogRepository extends JpaRepository<LoginLog, Long>{

}